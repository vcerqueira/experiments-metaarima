from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier

from src.meta.arima._base import MetaARIMAUtils
from src.meta.arima.meta_arima import MetaARIMA
from src.load_data.config import DATASETS
from src.config import (ORDER_MAX,
                        N_TRIALS_SPACE,
                        MAX_N_TRIALS,
                        QUANTILE_THR,
                        LAMBDA,
                        MMR,
                        BASE_OPTIM)

# data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
# data_name, group = 'M4', 'Weekly'
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, extended=True)

train, test = data_loader.train_test_split(df, horizon=horizon)

feats = pd.read_csv(f'assets/features/features,{data_name},{group}.csv')
input_variables = feats.set_index('unique_id').columns.tolist()

cv = pd.read_csv(f'assets/metadata_cv/arima,{data_name},{group}.csv')
cv = cv.merge(feats, on=['unique_id']).set_index('unique_id')

model_names = MetaARIMAUtils.get_models_sf(season_length=freq_int,
                                           return_names=True,
                                           max_config=ORDER_MAX)

X = cv.loc[:, input_variables].fillna(-1)
y = cv.loc[:, model_names]

kfcv = KFold(n_splits=5, random_state=1, shuffle=True)

results = []
for j, (train_index, test_index) in enumerate(kfcv.split(X)):
    print(f"Fold {j}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

    X_train = X.iloc[train_index, :]
    y_train = y.iloc[train_index, :]
    X_test = X.iloc[test_index, :]
    y_test = y.iloc[test_index, :]

    mod = ClassifierChain(LGBMClassifier(verbosity=-1))

    print('MetaARIMA fitting')
    meta_arima = MetaARIMA(model=mod,
                           freq=freq_str,
                           season_length=freq_int,
                           n_trials=MAX_N_TRIALS,
                           quantile_thr=QUANTILE_THR,
                           base_optim=BASE_OPTIM,
                           use_mmr=MMR,
                           mmr_lambda=LAMBDA)

    meta_arima.meta_fit(X_train, y_train)

    print('MetaARIMA inference')
    n_trials_preds = {}
    for n_trials_ in N_TRIALS_SPACE:
        print('N TRIALS', n_trials_)
        meta_arima.n_trials = n_trials_

        n_trials_preds[n_trials_] = meta_arima.meta_predict(X_test)

    print('MetaARIMA evaluating')
    for i, (uid, x) in enumerate(X_test.iterrows()):
        print(i, uid)
        df_uid = train.query(f'unique_id=="{uid}"')

        scores = {}
        for n_trials_ in n_trials_preds:
            uid_list = n_trials_preds[n_trials_]

            try:
                meta_arima.fit(df_uid, config_space=uid_list[i])
            except ValueError:
                scores[f'MetaARIMA({n_trials_})'] = np.nan
                continue

            err_meta_mmr = cv.loc[uid, meta_arima.selected_config]

            scores[f'MetaARIMA({n_trials_})'] = err_meta_mmr

        scores['unique_id'] = f'{data_name},{group},{uid}'
        scores['AutoARIMA'] = cv.loc[uid, 'score_AutoARIMA']

        pprint(scores)

        results.append(scores)

results_df = pd.DataFrame(results)

print(results_df.mean())
print(results_df.median())
print(results_df.rank(axis=1, na_option='bottom').mean())
print(results_df.dropna().mean())
print(results_df.dropna().median())
print(results_df.dropna().rank(axis=1).mean())
