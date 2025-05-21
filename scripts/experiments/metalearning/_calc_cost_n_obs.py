from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.load_data.config import DATASETS
from src.config import MMR, N_TRIALS, QUANTILE_THR, BASE_OPTIM, LAMBDA

data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
# data_name, group = 'M4', 'Weekly'
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

mod = ClassifierChain(LGBMClassifier(verbosity=-1))

meta_arima = MetaARIMA(model=mod,
                       freq=freq_str,
                       season_length=freq_int,
                       n_trials=N_TRIALS,
                       quantile_thr=QUANTILE_THR,
                       use_mmr=MMR,
                       base_optim=BASE_OPTIM,
                       mmr_lambda=LAMBDA)

meta_arima.meta_fit(X_train, y_train)
pred_list = meta_arima.meta_predict(X_test)

i = 0
uid = X_test.index[0]
x = X_test.iloc[0]
print(i, uid)

n_periods = 1000
start_date = pd.Timestamp('2000-01-01') + pd.DateOffset(months=np.random.randint(0, 120))
date_range = pd.date_range(end=start_date, periods=n_periods, freq='ME')
df_uid = pd.DataFrame({
    'unique_id': [uid] * n_periods,
    'ds': date_range,
    'y': np.random.randn(n_periods)
})

meta_arima.fit(df_uid, config_space=pred_list[i])
meta_arima.model.tot_nobs

1000 * len(pred_list[i])
1000 * 94

# 2736
# 36*len(pred_list[i]) + 72*(len(pred_list[i])/2) + 144*(len(pred_list[i])/4) + 288*(len(pred_list[i])/8)
