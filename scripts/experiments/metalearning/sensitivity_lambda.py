from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRFRegressor

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.load_data.config import DATASETS
from src.config import (LAMBDA_SPACE,
                        LAMBDA,
                        QUANTILE_THR,
                        N_TRIALS,
                        MMR,
                        BASE_OPTIM,
                        N_FOLDS,
                        RANDOM_SEED)

# data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
# data_name, group = 'M4', 'Weekly'
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, extended=True)

train, _ = data_loader.train_test_split(df, horizon=horizon)

mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=freq_int)

X, y, _, _, cv = mdr.read(fill_na_value=-1)
print(y.shape)
print(cv.shape)

kfcv = KFold(n_splits=N_FOLDS, random_state=RANDOM_SEED, shuffle=True)

results = []
for j, (train_index, test_index) in enumerate(kfcv.split(X)):
    print(f"Fold {j}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

    X_train = X.iloc[train_index, :]
    y_train = y.iloc[train_index, :]
    X_test = X.iloc[test_index, :]
    y_test = y.iloc[test_index, :]

    mod = XGBRFRegressor()

    print('MetaARIMA fitting')
    meta_arima = MetaARIMA(model=mod,
                           freq=freq_str,
                           season_length=freq_int,
                           n_trials=N_TRIALS,
                           quantile_thr=QUANTILE_THR,
                           base_optim=BASE_OPTIM,
                           use_mmr=MMR,
                           mmr_lambda=LAMBDA)

    meta_arima.meta_fit(X_train, y_train)

    print('MetaARIMA inference')
    lambda_preds = {}
    for lambda_ in LAMBDA_SPACE:
        print('LAMBDA_SPACE', lambda_)
        meta_arima.mmr_lambda = lambda_

        lambda_preds[lambda_] = meta_arima.meta_predict(X_test)

    print('MetaARIMA evaluating')
    for i, (uid, x) in enumerate(X_test.iterrows()):
        print(i, uid)
        df_uid = train.query(f'unique_id=="{uid}"')

        scores = {}
        for lambda__ in lambda_preds:
            uid_list = lambda_preds[lambda__]

            try:
                meta_arima.fit(df_uid, config_space=uid_list[i])
            except ValueError:
                scores[f'MetaARIMA({str(lambda__)})'] = np.nan
                continue

            err_meta_mmr = cv.loc[uid, meta_arima.selected_config]

            scores[f'MetaARIMA({lambda__})'] = err_meta_mmr

        scores['unique_id'] = f'{data_name},{group},{uid}'
        scores['AutoARIMA'] = cv.loc[uid, 'score_AutoARIMA']

        pprint(scores)

        results.append(scores)

results_df = pd.DataFrame(results)

results_df.to_csv(f'assets/results/sensitivity/lambda,{data_name},{group}.csv', index=False)

print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))
print(results_df.rank(axis=1, na_option='bottom').mean())
print(results_df.dropna().mean())
print(results_df.dropna().median())
print(results_df.dropna().rank(axis=1).mean())
