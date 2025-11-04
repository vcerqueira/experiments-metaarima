import re
from pprint import pprint

import pandas as pd
import numpy as np

from src.load_data.config import DATASETS
from src.meta.arima._data_reader import ModelIO
from src.meta.arima._data_reader import MetadataReader

FILENAME = 'assets/trained_metaarima_m4m.joblib.gz'

meta_arima = ModelIO.load_model(FILENAME)

data_name, group = 'M3', 'Quarterly'

tgt_data_loader = DATASETS[data_name]
tgt_df, tgt_horizon, tgt_n_lags, tgt_freq_str, tgt_freq_int = tgt_data_loader.load_everything(group, extended=True)
tgt_train, _ = tgt_data_loader.train_test_split(tgt_df, horizon=tgt_horizon)
tgt_mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=tgt_freq_int)
tgt_X, _, _, _, tgt_cv = tgt_mdr.read(fill_na_value=-1)

pred_list = meta_arima.meta_predict(tgt_X)

results = []
for i, (uid, x) in enumerate(tgt_X.iterrows()):
    # uid, x = tgt_X.iloc[0].name, tgt_X.iloc[0]
    print(i, uid)

    df_uid = tgt_train.query(f'unique_id=="{uid}"').copy()

    try:
        meta_arima.fit(df_uid, config_space=pred_list[i])
    except ValueError:
        continue

    auto_arima_config = tgt_cv.loc[uid, 'auto_config']
    meta_arima.selected_config = re.sub(r'\[[^\]]*\]', f'[{tgt_freq_int}]', meta_arima.selected_config)

    err_meta = tgt_cv.loc[uid, meta_arima.selected_config]
    err_auto = tgt_cv.loc[uid, 'score_AutoARIMA']
    err_snaive = tgt_cv.loc[uid, 'score_SeasNaive']
    err_theta = tgt_cv.loc[uid, 'score_AutoTheta']
    err_ets = tgt_cv.loc[uid, 'score_AutoETS']

    config212 = f'ARIMA(2,1,2)(0,0,0)[{tgt_freq_int}]'
    config100 = f'ARIMA(1,0,0)(0,0,0)[{tgt_freq_int}]'

    err_arima212 = tgt_cv.loc[uid, config212]
    err_arima100 = tgt_cv.loc[uid, config100]

    comp = {
        'MetaARIMA': err_meta,
        'AutoARIMA': err_auto,
        'ARIMA(2,1,2)': err_arima212,
        'ARIMA(1,0,0)': err_arima100,
        'SeasonalNaive': err_snaive,
        'AutoTheta': err_theta,
        'AutoETS': err_ets,
        'unique_id': uid,
        'dataset': f'{data_name}({group})',
    }

    pprint(comp)
    results_df = pd.DataFrame(results)
    print(results_df.mean(numeric_only=True))

    results.append(comp)

results_df = pd.DataFrame(results)
print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))
