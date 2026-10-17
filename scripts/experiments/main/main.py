import argparse
import os
from pprint import pprint
from copy import deepcopy

import pandas as pd

from utilsforecast.losses import mase
from statsforecast import StatsForecast
from statsforecast.models import (AutoARIMA,
                                  AutoETS,
                                  AutoTheta,
                                  SeasonalNaive,
                                  ARIMA)

from src.meta.arima._data_reader import ModelIO
from src.chronos_data import ChronosDataset

parser = argparse.ArgumentParser(description="Evaluate a saved MetaARIMA model against statistical baselines.")
parser.add_argument("--source", default="m4_monthly", help="M4 dataset of the saved metamodel.")
parser.add_argument("--target", default="monash_car_parts", help="Dataset to forecast.")
args = parser.parse_args()

source = args.source
FILENAME = f'assets/trained_metaarima_{source}_catboost.joblib.gz'
meta_arima = ModelIO.load_model(FILENAME)

target = args.target
df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
train, test = ChronosDataset.time_wise_split(df, horizon)

sf_models = [AutoARIMA(season_length=seas_len),
             AutoETS(season_length=seas_len),
             AutoTheta(season_length=seas_len),
             SeasonalNaive(season_length=seas_len),
             ARIMA(order=(1, 0, 0), season_length=seas_len, alias='AR'),
             ARIMA(order=(0, 0, 1), season_length=seas_len, alias='MA'),
             ARIMA(order=(1, 0, 1), season_length=seas_len, alias='ARMA'),
             ARIMA(order=(2, 1, 2), season_length=seas_len, alias='ARIMA'),
             ARIMA(order=(1, 0, 1),
                   seasonal_order=(1, 0, 1),
                   season_length=seas_len,
                   alias='SARIMA')
             ]

model_names = ['MetaARIMA', 'AutoARIMA',
               'AutoETS', 'AutoTheta',
               'AR', 'MA', 'ARMA',
               'ARIMA', 'SeasonalNaive']

uids = train['unique_id'].unique().tolist()

results = []
for uid in uids:
    print(uid)
    # uid="T000055"

    df_uid_tr = train.query(f'unique_id=="{uid}"').reset_index(drop=True)
    df_uid_ts = test.query(f'unique_id=="{uid}"').reset_index(drop=True)
    if df_uid_ts.isna().any()['y']:
        continue

    try:
        meta_arima.fit(df_uid_tr, freq=freq, seas_length=seas_len)
        fcst_ma = meta_arima.predict(h=horizon)
        sf = StatsForecast(models=deepcopy(sf_models), freq=freq)
        sf.fit(df_uid_tr)
        fcst_aa = sf.forecast(h=horizon)
    except NotImplementedError:
        print(f"skipping {uid}: series too short for AutoETS")
        continue

    uid_test = df_uid_ts.merge(fcst_ma, on=['unique_id', 'ds'])
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

os.makedirs('assets/results/main', exist_ok=True)
results_df.to_csv(f'assets/results/main/{target}.csv', index=False)
