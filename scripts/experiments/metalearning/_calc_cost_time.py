import multiprocessing
import time

multiprocessing.set_start_method('fork', force=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRFRegressor
from tsfeatures import tsfeatures
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.load_data.config import DATASETS
from src.config import MMR, N_TRIALS, QUANTILE_THR, BASE_OPTIM, LAMBDA

data_name, group = 'M3', 'Monthly'
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, extended=True)

train, _ = data_loader.train_test_split(df, horizon=horizon)

mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=freq_int)

X, y, _, _, cv = mdr.read(fill_na_value=-1)

# note that this is cv on the time series set (80% of time series for train, 20% for testing)
# partition is done at time series level, not in time dimension
kfcv = KFold(n_splits=5, random_state=1, shuffle=True)

results = []
j, (train_index, test_index) = next(enumerate(kfcv.split(X)))
print(f"Fold {j}:")
print(f"  Train: index={train_index}")
print(f"  Test:  index={test_index}")

X_train = X.iloc[train_index, :]
y_train = y.iloc[train_index, :]
X_test = X.iloc[test_index, :]
y_test = y.iloc[test_index, :]

metaarima_fit_start = time.time()
meta_arima = MetaARIMA(model=XGBRFRegressor(),
                       freq=freq_str,
                       season_length=freq_int,
                       n_trials=N_TRIALS,
                       quantile_thr=QUANTILE_THR,
                       use_mmr=MMR,
                       base_optim=BASE_OPTIM,
                       mmr_lambda=LAMBDA)

meta_arima.meta_fit(X_train, y_train)
metaarima_fit_time = time.time() - metaarima_fit_start

test_uids = X_test.index.tolist()

metaarima_start = time.time()
for i, uid in enumerate(test_uids):
    print(i, uid)

    df_uid = train.query(f'unique_id=="{uid}"').copy()

    feat_df = tsfeatures(df_uid, freq=freq_int).set_index('unique_id')

    uid_configs = meta_arima.meta_predict(feat_df.fillna(-1))[0]

    meta_arima.fit(df_uid, config_space=uid_configs)

    fcst = meta_arima.model.sf.predict(h=horizon)

    # try:
    #     meta_arima.fit(df_uid, config_space=uid_configs)
    # except ValueError:
    #     continue

metaarima_time = time.time() - metaarima_start

metaarima_start = time.time()
df_uids = train.query(f'unique_id==@test_uids').copy()
feat_df = tsfeatures(df_uids, freq=freq_int).set_index('unique_id')
uid_configs = meta_arima.meta_predict(feat_df.fillna(-1))

for i, uid in enumerate(test_uids):
    print(i, uid)

    df_uid = train.query(f'unique_id=="{uid}"').copy()

    meta_arima.fit(df_uid, config_space=uid_configs[i])

    fcst = meta_arima.model.sf.predict(h=horizon)

metaarima_time = time.time() - metaarima_start

autoarima_start = time.time()
for i, uid in enumerate(test_uids):
    print(i, uid)

    df_uid = train.query(f'unique_id=="{uid}"').copy()

    sf_auto = StatsForecast(models=[AutoARIMA()], freq=freq_str)
    sf_auto.fit(df=df_uid)
    fcst_auto = sf_auto.predict(h=horizon)
autoarima_time = time.time() - autoarima_start

print(f"\nMetaARIMA time: {metaarima_time:.2f}s")
print(f"\nMetaARIMA fit time: {metaarima_fit_time:.2f}s")
print(f"AutoARIMA time: {autoarima_time:.2f}s")
