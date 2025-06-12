import re
from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.load_data.config import DATASETS
from src.config import MMR, N_TRIALS, QUANTILE_THR, BASE_OPTIM, LAMBDA

# data_name, group = 'M3', 'Monthly'

target_sets = [
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
    # ('M4', 'Monthly'),
    ('M4', 'Quarterly'),
]

# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
# data_name, group = 'M4', 'Quarterly'
# print(data_name, group)

# -- train metamodel

source_data_name, source_group = 'M3', 'Monthly'
data_loader = DATASETS[source_data_name]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(source_group, extended=True)
train, _ = data_loader.train_test_split(df, horizon=horizon)
mdr = MetadataReader(dataset_name=source_data_name, group=source_group, freq_int=freq_int)
X, y, _, _, cv = mdr.read(fill_na_value=-1)

mod = ClassifierChain(LGBMClassifier(verbosity=-1))

meta_arima = MetaARIMA(model=mod,
                       freq=freq_str,
                       season_length=freq_int,
                       n_trials=N_TRIALS,
                       quantile_thr=QUANTILE_THR,
                       use_mmr=MMR,
                       base_optim=BASE_OPTIM,
                       mmr_lambda=LAMBDA)

meta_arima.meta_fit(X, y)
# meta_arima.meta_fit(X.head(700), y.head(700))

results = []
#for j, (train_index, test_index) in enumerate(kfcv.split(X)):
for j, (data_name, group) in enumerate(target_sets):
    # data_name, group = 'Tourism', 'Quarterly'
    print(data_name, group)

    # read sets...
    tgt_data_loader = DATASETS[data_name]
    tgt_df, tgt_horizon, tgt_n_lags, tgt_freq_str, tgt_freq_int = tgt_data_loader.load_everything(group, extended=True)
    tgt_train, _ = tgt_data_loader.train_test_split(tgt_df, horizon=tgt_horizon)
    tgt_mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=tgt_freq_int)
    tgt_X, tgt_y, _, _, tgt_cv = tgt_mdr.read(fill_na_value=-1)

    pred_list = meta_arima.meta_predict(tgt_X)

    for i, (uid, x) in enumerate(tgt_X.iterrows()):
        # uid, x = tgt_X.iloc[0].name, tgt_X.iloc[0]
        print(i, uid)

        df_uid = tgt_train.query(f'unique_id=="{uid}"').copy()
        meta_arima.selected_config = re.sub(r'\[[^\]]*\]', f'[{tgt_freq_int}]', meta_arima.selected_config)

        try:
            meta_arima.fit(df_uid, config_space=pred_list[i])
        except ValueError:
            continue

        auto_arima_config = tgt_cv.loc[uid, 'auto_config']

        err_meta = tgt_cv.loc[uid, meta_arima.selected_config]
        err_auto = tgt_cv.loc[uid, 'score_AutoARIMA']
        err_snaive = tgt_cv.loc[uid, 'score_SeasNaive']
        err_theta = tgt_cv.loc[uid, 'score_AutoTheta']
        err_ets = tgt_cv.loc[uid, 'score_AutoETS']

        config212 = f'ARIMA(2,1,2)(0,0,0)[{tgt_freq_int}]'
        config2121 = f'ARIMA(2,1,2)(1,0,0)[{tgt_freq_int}]'
        config100 = f'ARIMA(1,0,0)(0,0,0)[{tgt_freq_int}]'

        err_arima212 = tgt_cv.loc[uid, config212]
        err_arima2121 = tgt_cv.loc[uid, config2121]
        err_arima100 = tgt_cv.loc[uid, config100]

        try:
            err_auto2 = tgt_cv.loc[uid, auto_arima_config]
        except KeyError:
            err_auto2 = np.nan

        comp = {
            'MetaARIMA': err_meta,
            'AutoARIMA': err_auto,
            'AutoARIMA2': err_auto2,
            'ARIMA(2,1,2)': err_arima212,
            'ARIMA(2,1,2)(1,0,0)': err_arima2121,
            'ARIMA(1,0,0)': err_arima100,
            'SeasonalNaive': err_snaive,
            'AutoTheta': err_theta,
            'AutoETS': err_ets,
            'unique_id': uid,
            'dataset': f'{data_name}({group})',
        }

        pprint(comp)

        results.append(comp)

results_df = pd.DataFrame(results)
results_df.to_csv(f'assets/results/sensitivity/transfer,{source_data_name},{source_group}.csv', index=False)
