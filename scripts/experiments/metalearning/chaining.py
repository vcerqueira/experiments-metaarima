from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain

import xgboost as xgb

from src.arima.meta import MetaARIMAUtils, MetaARIMABase
from src.load_data.config import DATASETS, DATA_GROUPS, GROUP_IDX

data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)

train, test = data_loader.train_test_split(df, horizon=horizon)

cv = pd.read_csv('assets/metadata_cv/arima_cv2_M3,Monthly.csv')
feats = pd.read_csv('assets/features/train_feature_set_M3,Monthly.csv')

cv = cv.merge(feats, on=['unique_id']).set_index('unique_id')

input_variables = feats.set_index('unique_id').columns.tolist()

model_names = MetaARIMAUtils.get_models_sf(season_length=12, return_names=True)

X = cv.loc[:, input_variables].fillna(-1)
y = cv.loc[:, model_names].apply(lambda x: (x <= x.quantile(.05)).astype(int), axis=1)


cv.loc[:, model_names].corr()
# max marginal rel
# add avg rank on meta-predictors
## only makes sense with a ranking approach

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

mod = ClassifierChain(xgb.XGBRFClassifier(n_estimators=100))
# mod = xgb.XGBRFClassifier(n_estimators=100)
# mod = ClassifierChain(xgb.XGBClassifier())
mod.fit(X_train, y_train)

preds = pd.DataFrame(mod.predict_proba(X_test), columns=model_names)
preds_list = preds.apply(lambda x: x.sort_values().index[:20].tolist(), axis=1)

# preds_list[0]
# X_test.iloc[0].name



sc = []
for i in range(len(preds_list)):
    print(i)

    uid = X_test.iloc[i].name
    df_uid = train.query(f'unique_id=="{uid}"')

    mod2 = MetaARIMABase(configs=preds_list[i], freq='ME', season_length=12)
    mod2.fit(df_uid)
    config_selected = MetaARIMAUtils.get_model_order(mod2.sf.fitted_[0][0].model_, as_alias=True, alias_freq=12)

    auto_arima_config = cv.loc[uid, 'auto_config']

    err_meta = cv.loc[uid, config_selected]
    err_auto = cv.loc[uid, 'score_AutoARIMA']
    try:
        err_auto2 = cv.loc[uid, auto_arima_config]
    except KeyError:
        err_auto2 = np.nan

    comp = {'meta': err_meta, 'auto': err_auto, 'auto2': err_auto2}

    sc.append(comp)

pd.DataFrame(sc)

print(pd.DataFrame(sc).mean())
print(pd.DataFrame(sc).median())
print(pd.DataFrame(sc).rank(axis=1).mean())
print(pd.DataFrame(sc).dropna().mean())
print(pd.DataFrame(sc).dropna().median())
print(pd.DataFrame(sc).dropna().rank(axis=1).mean())

print(pd.DataFrame(sc).drop(columns='auto2').mean())
print(pd.DataFrame(sc).drop(columns='auto2').median())
print(pd.DataFrame(sc).drop(columns='auto2').rank(axis=1).mean())



r = pd.DataFrame(sc)
100 * ((r['meta'] - r['auto']) / r['auto'])
(100 * ((r['meta'] - r['auto']) / r['auto'])).median()

(r['meta'] - r['auto']).mean()
