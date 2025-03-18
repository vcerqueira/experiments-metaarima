from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain

import xgboost as xgb

from src.arima.meta import MetaARIMAUtils, MetaARIMA
from src.load_data.config import DATASETS

# data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
# data_name, group = 'M4', 'Weekly'
print(data_name, group)
data_loader = DATASETS[data_name]

TEST_SIZE_UIDS = 0.1
N_TRIALS = 20
QUANTILE_THR = 0.1
N_ESTIMATORS = 25
MMR = True

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, extended=True)

train, test = data_loader.train_test_split(df, horizon=horizon)

cv = pd.read_csv(f'assets/metadata_cv/arima,{data_name},{group}.csv')
feats = pd.read_csv(f'assets/features/features,{data_name},{group}.csv')

cv = cv.merge(feats, on=['unique_id']).set_index('unique_id')

input_variables = feats.set_index('unique_id').columns.tolist()

model_names = MetaARIMAUtils.get_models_sf(season_length=freq_int, return_names=True)

X = cv.loc[:, input_variables].fillna(-1)
y = cv.loc[:, model_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE_UIDS)

mod = ClassifierChain(xgb.XGBRFClassifier(n_estimators=N_ESTIMATORS))

meta_arima = MetaARIMA(model=mod,
                       freq=freq_str,
                       season_length=freq_int,
                       n_trials=N_TRIALS,
                       quantile_thr=QUANTILE_THR,
                       use_mmr=MMR,
                       mmr_lambda=0.5)

meta_arima.meta_fit(X_train, y_train)

# lambda_space = np.linspace(start=0, stop=1, num=21)
lambda_space = np.linspace(start=0, stop=1, num=11)

lambda_pred_list = {}
for lambda_ in lambda_space:
    print(lambda_)
    meta_arima.mmr_lambda = lambda_

    lambda_pred_list[lambda_] = meta_arima.meta_predict(X_test)

results = []
for i, (uid, x) in enumerate(X_test.iterrows()):
    print(i, uid)
    # if i > 5:
    #     break

    df_uid = train.query(f'unique_id=="{uid}"')

    scores = {}
    for lambda_v in lambda_pred_list:
        uid_list = lambda_pred_list[lambda_v]

        meta_arima.fit(df_uid, config_list=uid_list[i])
        mod_ = meta_arima.model.sf.fitted_[0][0]
        config_selected_mrr = MetaARIMAUtils.get_model_order(mod_.model_,
                                                             as_alias=True,
                                                             alias_freq=freq_int)

        auto_arima_config = cv.loc[uid, 'auto_config']

        err_meta_mmr = cv.loc[uid, config_selected_mrr]

        scores[f'MetaARIMA({np.round(lambda_v, 2)})'] = err_meta_mmr

    scores['AutoARIMA'] = cv.loc[uid, 'score_AutoARIMA']

    pprint(scores)

    results.append(scores)

results_df = pd.DataFrame(results)

print(results_df.mean())
print(results_df.median())
print(results_df.rank(axis=1).mean())
print(results_df.dropna().mean())
print(results_df.dropna().median())
print(results_df.dropna().rank(axis=1).mean())
