from pprint import pprint
from copy import deepcopy

import pandas as pd

from utilsforecast.losses import mase
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive

from src.meta.arima._data_reader import ModelIO
from src.chronos_data import ChronosDataset

OVERRIDE_DS = False
algorithm = 'catboost'
source = 'm4_monthly'
FILENAME = f'assets/trained_metaarima_{source}_{algorithm}.joblib.gz'
meta_arima = ModelIO.load_model(FILENAME)

target = 'monash_m1_monthly'

df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
train, test = ChronosDataset.time_wise_split(df, horizon)

sf_models = [AutoARIMA(season_length=seas_len),
             SeasonalNaive(season_length=seas_len)]

#  TODO ADD TSFM'S NAMES TO LIST
model_names = ['MetaARIMA', 'AutoARIMA', 'SeasonalNaive']

uids = train['unique_id'].unique().tolist()

results, predictions = [], []
for uid in uids:
    print(uid)
    # uid="T000055"

    df_uid_tr = train.query(f'unique_id=="{uid}"').reset_index(drop=True)
    df_uid_ts = test.query(f'unique_id=="{uid}"').reset_index(drop=True)
    if df_uid_ts.isna().any()['y']:
        continue

    meta_arima.fit(df_uid_tr, freq=freq, seas_length=seas_len)

    fcst_ma = meta_arima.predict(h=horizon)

    sf = StatsForecast(models=deepcopy(sf_models), freq=freq)
    sf.fit(df_uid_tr)

    fcst_aa = sf.forecast(h=horizon)

    # TODO tsfm inference
    fcst_tsfm1 = tsfm.predict(h=horizon)
    # TODO transform to structure like fcst_aa

    if OVERRIDE_DS:
        fcst_ma['ds'] = df_uid_ts['ds'].values
        fcst_aa['ds'] = df_uid_ts['ds'].values
        # TODO override values
        fcst_tsfm1['ds'] = fcst_tsfm1['ds'].values

    uid_test = df_uid_ts.merge(fcst_ma, on=['unique_id', 'ds'])
    uid_test = uid_test.merge(fcst_aa, on=['unique_id', 'ds'])
    # TODO add to uid_test
    uid_test = uid_test.merge(fcst_tsfm1, on=['unique_id', 'ds'])

    err = mase(df=uid_test, models=model_names, seasonality=seas_len, train_df=df_uid_tr)

    pprint(err)

    predictions.append(uid_test)
    results.append(err)
    results_df = pd.concat(results)
    print(results_df.mean(numeric_only=True))
    print(results_df.median(numeric_only=True))

results_df = pd.concat(results)
predictions_df = pd.concat(predictions).reset_index(drop=True)
print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))

results_df.to_csv(f'assets/results/tsfm/scores,{target}.csv', index=False)
predictions_df.to_csv(f'assets/results/tsfm/predictions,{target}.csv', index=False)
