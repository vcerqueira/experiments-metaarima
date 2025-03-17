from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain

import xgboost as xgb

from src.arima.meta import MetaARIMAUtils, MetaARIMA
from src.load_data.config import DATASETS

# data_name, group = 'M3', 'Monthly'
data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
# data_name, group = 'M4', 'Quarterly'
print(data_name, group)
data_loader = DATASETS[data_name]

TEST_SIZE_UIDS = 0.2
N_TRIALS = 20
QUANTILE_THR = 0.15

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)

train, test = data_loader.train_test_split(df, horizon=horizon)

cv = pd.read_csv(f'assets/metadata_cv/arima,{data_name},{group}.csv')
feats = pd.read_csv(f'assets/features/features,{data_name},{group}.csv')

cv = cv.merge(feats, on=['unique_id']).set_index('unique_id')

input_variables = feats.set_index('unique_id').columns.tolist()

model_names = MetaARIMAUtils.get_models_sf(season_length=freq_int, return_names=True)

X = cv.loc[:, input_variables].fillna(-1)
y = cv.loc[:, model_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE_UIDS)


y_train.corr()



# mod = ClassifierChain(xgb.XGBClassifier(n_estimators=100))
# mod = xgb.XGBRFClassifier(n_estimators=100)
# mod = ClassifierChain(xgb.XGBClassifier())
mod = ClassifierChain(xgb.XGBRFClassifier(n_estimators=100))

meta_arima = MetaARIMA(model=mod,
                       freq=freq_str,
                       season_length=freq_int,
                       n_trials=N_TRIALS,
                       quantile_thr=QUANTILE_THR)

meta_arima.meta_fit(X_train, y_train)

pred_list = meta_arima.meta_predict(X_test)

results = []
for i, (uid, x) in enumerate(X_test.iterrows()):
    print(i, uid)

    df_uid = train.query(f'unique_id=="{uid}"')

    meta_arima.fit(df_uid, config_list=pred_list.values[i])

    mod_ = meta_arima.model.sf.fitted_[0][0]

    config_selected = MetaARIMAUtils.get_model_order(mod_.model_,
                                                     as_alias=True,
                                                     alias_freq=freq_int)
    auto_arima_config = cv.loc[uid, 'auto_config']

    err_meta = cv.loc[uid, config_selected]
    err_auto = cv.loc[uid, 'score_AutoARIMA']
    try:
        err_auto2 = cv.loc[uid, auto_arima_config]
    except KeyError:
        err_auto2 = np.nan

    comp = {'MetaARIMA': err_meta,
            'AutoARIMA': err_auto,
            'AutoARIMA2': err_auto2,  # what is this?
            }

    results.append(comp)

results_df = pd.DataFrame(results)

print(results_df.mean())
print(results_df.median())
print(results_df.rank(axis=1).mean())
print(results_df.dropna().mean())
print(results_df.dropna().median())
print(results_df.dropna().rank(axis=1).mean())
print(results_df.drop(columns='AutoARIMA2').mean())
print(results_df.drop(columns='AutoARIMA2').median())
print(results_df.drop(columns='AutoARIMA2').rank(axis=1).mean())
