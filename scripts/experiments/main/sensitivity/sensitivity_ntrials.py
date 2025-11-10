import copy
from pprint import pprint
from copy import deepcopy

import pandas as pd

from utilsforecast.losses import mase
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from src.meta.arima._data_reader import ModelIO
from src.chronos_data import ChronosDataset
from src.config import N_TRIALS_SPACE

OVERRIDE_DS = False
algorithm = 'catboost'
source = 'm4_monthly'
FILENAME = f'assets/trained_metaarima_{source}_{algorithm}.joblib.gz'
meta_arima = ModelIO.load_model(FILENAME)

target = 'monash_m3_monthly'
# target = 'monash_hospital'
df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
train, test = ChronosDataset.time_wise_split(df, horizon)

sf_models = [AutoARIMA(season_length=seas_len)]

uids = train['unique_id'].unique().tolist()

meta_arima_d = {}
for n_trials_ in N_TRIALS_SPACE:
    print('n_trials_:', n_trials_)
    m_a = copy.deepcopy(meta_arima)

    m_a.n_trials = n_trials_

    meta_arima_d[f'MetaARIMA({n_trials_})'] = m_a

model_names = [*meta_arima_d] + ['AutoARIMA']

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

    sf = StatsForecast(models=deepcopy(sf_models), freq=freq)
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

results_df.to_csv(f'assets/results/main/ntrials,{target}.csv', index=False)
