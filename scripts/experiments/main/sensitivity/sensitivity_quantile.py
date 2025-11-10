import copy
from pprint import pprint

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from utilsforecast.losses import mase
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.chronos_data import ChronosDataset
from src.config import (MMR,
                        N_TRIALS,
                        BASE_OPTIM,
                        LAMBDA,
                        QUANTILE_SPACE,
                        PCA_N_COMPONENTS,
                        ORDER_MAX,
                        BEST_CATBOOST_PARAMS)

# -- train metamodel
algorithm = 'catboost'
source = 'm4_monthly'
FILENAME = f'assets/trained_metaarima_{source}_{algorithm}.joblib.gz'
_, _, _, freq_str, freq_int = ChronosDataset.load_everything(source)

mdr = MetadataReader(group=source, freq_int=freq_int)
X, y, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1, max_config=ORDER_MAX)

model = CatBoostRegressor(**BEST_CATBOOST_PARAMS[source])

meta_arima_d = {}
for qtl in QUANTILE_SPACE:
    print('qtl:', qtl)
    meta_arima_ = MetaARIMA(model=model,
                            freq=freq_str,
                            season_length=freq_int,
                            n_trials=N_TRIALS,
                            quantile_thr=qtl,
                            pca_n_components=PCA_N_COMPONENTS,
                            use_mmr=MMR,
                            base_optim=BASE_OPTIM,
                            mmr_lambda=LAMBDA)

    meta_arima_.meta_fit(X, y)

    meta_arima_d[f'MetaARIMA({np.round(qtl, 2)})'] = meta_arima_

model_names = [*meta_arima_d] + ['AutoARIMA']

target = 'monash_m3_monthly'
# target = 'monash_hospital'
df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
train, test = ChronosDataset.time_wise_split(df, horizon)

sf_models = [AutoARIMA(season_length=seas_len)]

uids = train['unique_id'].unique().tolist()

results = []
for uid in uids:
    print(uid)

    df_uid_tr = train.query(f'unique_id=="{uid}"').reset_index(drop=True)
    df_uid_ts = test.query(f'unique_id=="{uid}"').reset_index(drop=True)

    meta_arima_fcst = {}
    for ma_k, ma_v in meta_arima_d.items():
        # print(ma_k)
        ma_v.fit(df_uid_tr, freq=freq, seas_length=seas_len)

        fcst_ma_ = ma_v.predict(h=horizon)

        meta_arima_fcst[ma_k] = fcst_ma_['MetaARIMA'].values

    fcst_meta_arima = pd.DataFrame(meta_arima_fcst)

    sf = StatsForecast(models=copy.deepcopy(sf_models), freq=freq)
    sf.fit(df_uid_tr)

    fcst_aa = sf.forecast(h=horizon).reset_index()
    fcst_meta_arima['ds'] = fcst_aa['ds']
    fcst_meta_arima['unique_id'] = fcst_aa['unique_id']

    uid_test = df_uid_ts.merge(fcst_meta_arima, on=['unique_id', 'ds'])
    uid_test = uid_test.merge(fcst_aa, on=['unique_id', 'ds'])

    err = mase(df=uid_test, models=model_names, seasonality=seas_len, train_df=df_uid_tr)

    pprint(err)

    results.append(err)
    results_df = pd.concat(results)
    print(results_df.mean(numeric_only=True))
    print(results_df.median(numeric_only=True))

results_df = pd.concat(results)
print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))

results_df.to_csv(f'assets/results/main/quantile,{target}.csv', index=False)
