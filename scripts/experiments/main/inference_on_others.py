from pprint import pprint

import pandas as pd
import numpy as np

from datasetsforecast.m3 import M3
from utilsforecast.losses import mase
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from src.meta.arima._data_reader import ModelIO, MetadataReader
from src.chronos_data import ChronosDataset

FILENAME = 'assets/trained_metaarima_m4m.joblib.gz'
# FILENAME = 'assets/trained_metaarima_m4m_cb.joblib.gz'
meta_arima = ModelIO.load_model(FILENAME)

target = 'monash_m1_monthly'
df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)

sf = StatsForecast(models=[AutoARIMA(season_length=freq)], freq=freq)

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

    test = df_uid_ts.merge(fcst_ma, on=['unique_id', 'ds'])
    test = test.merge(fcst_aa_uid, on=['unique_id', 'ds'])

    err = mase(df=test, models=['MetaARIMA', 'AutoARIMA'],
               seasonality=freq, train_df=df_uid_tr)

    pprint(err)

    results.append(err)
    results_df = pd.concat(results)
    print(results_df.mean(numeric_only=True))
    print(results_df.median(numeric_only=True))

results_df = pd.concat(results)
print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))

results_df = results_df.merge(m3_monthly['unique_id'].value_counts(), on=['unique_id'])

results_df.query('count<100').mean(numeric_only=True)
