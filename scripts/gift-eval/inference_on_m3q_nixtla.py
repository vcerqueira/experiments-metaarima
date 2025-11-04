import re
from pprint import pprint

import pandas as pd
import numpy as np

from datasetsforecast.m3 import M3
from utilsforecast.losses import smape
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from src.meta.arima._data_reader import ModelIO

FILENAME = 'assets/trained_metaarima_m4m.joblib.gz'
meta_arima = ModelIO.load_model(FILENAME)

m3_quarterly, _, _ = M3.load(directory='./', group='Quarterly')

# Split train/test sets (last 8 observations for test)
horizon = 8
freq=4
freq_str='Q'
m3_test = m3_quarterly.groupby('unique_id', as_index=False).tail(horizon)
m3_train = m3_quarterly.drop(m3_test.index)

# uid_sample = ['Q1', 'Q2','Q99']
# m3_train = m3_train.query('unique_id==@uid_sample').reset_index(drop=True)

sf = StatsForecast(
    models=[AutoARIMA(season_length=freq)],
    freq=freq_str,
    n_jobs=1
)

sf.fit(m3_train)

fcst_aa = sf.forecast(h=horizon)

uids = m3_train['unique_id'].unique().tolist()

results = []
for uid in uids:
    print(uid)

    df_uid_tr = m3_train.query(f'unique_id=="{uid}"').reset_index(drop=True)
    df_uid_ts = m3_test.query(f'unique_id=="{uid}"').reset_index(drop=True)

    meta_arima.fit_model(df_uid_tr, freq=freq)

    fcst_ma = meta_arima.predict(h=horizon)
    fcst_aa_uid = fcst_aa.query(f'unique_id=="{uid}"')

    fcst_ma['ds'] = fcst_aa_uid['ds']

    test = df_uid_ts.merge(fcst_ma, on=['unique_id', 'ds'])
    test = test.merge(fcst_aa_uid, on=['unique_id', 'ds'])

    err = smape(df=test, models=['MetaARIMA', 'AutoARIMA'])

    pprint(err)

    results.append(err)
    results_df = pd.concat(results)
    print(results_df.mean(numeric_only=True))

results_df = pd.concat(results)
print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))


results_df = results_df.merge(m3_quarterly['unique_id'].value_counts(), on=['unique_id'])

results_df.query('count<40').mean(numeric_only=True)

