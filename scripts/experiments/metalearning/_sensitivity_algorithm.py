from pprint import pprint

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBRFClassifier
from sklearn.linear_model import RidgeClassifier

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._base import MetaARIMAUtils
from src.load_data.config import DATASETS
from src.config import N_TRIALS, QUANTILE_THR, BASE_OPTIM, LAMBDA

# data_name, group = 'M3', 'Monthly'
data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
# data_name, group = 'M4', 'Weekly'
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, extended=True)

train, _ = data_loader.train_test_split(df, horizon=horizon)

feats = pd.read_csv(f'assets/features/features,{data_name},{group}.csv')
cv = pd.read_csv(f'assets/metadata_cv/arima,{data_name},{group}.csv')
cv = cv.merge(feats, on=['unique_id']).set_index('unique_id')

input_variables = feats.set_index('unique_id').columns.tolist()
model_names = MetaARIMAUtils.get_models_sf(season_length=freq_int, return_names=True)

X = cv.loc[:, input_variables].fillna(-1)
y = cv.loc[:, model_names]

# note that this is cv on the time series set (80% of time series for train, 20% for testing)
# partition is done at time series level, not in time dimension
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

    params = {'colsample_bytree': 0.8351497755908628, 'learning_rate': 0.012652992231973307, 'max_depth': 3, 'min_child_samples': 69, 'n_estimators': 113, 'num_leaves': 28, 'subsample': 0.7047898756660642}

    mod1 = ClassifierChain(LGBMClassifier(verbosity=-1))
    mod2 = ClassifierChain(LGBMClassifier(verbosity=-1,**params))
    # mod2 = ClassifierChain(XGBRFClassifier())
    # mod3 = ClassifierChain(RidgeClassifier())
    # mod4 = XGBRFClassifier()

    meta_arima1 = MetaARIMA(model=mod1,
                           freq=freq_str,
                           season_length=freq_int,
                           n_trials=N_TRIALS,
                           quantile_thr=QUANTILE_THR,
                           use_mmr=True,
                           base_optim=BASE_OPTIM,
                           mmr_lambda=LAMBDA)

    meta_arima2 = MetaARIMA(model=mod2,
                            freq=freq_str,
                            season_length=freq_int,
                            n_trials=N_TRIALS,
                            quantile_thr=QUANTILE_THR,
                            use_mmr=True,
                            base_optim=BASE_OPTIM,
                            mmr_lambda=LAMBDA)

    # meta_arima3 = MetaARIMA(model=mod3,
    #                         freq=freq_str,
    #                         season_length=freq_int,
    #                         n_trials=N_TRIALS,
    #                         quantile_thr=QUANTILE_THR,
    #                         use_mmr=True,
    #                         base_optim=BASE_OPTIM,
    #                         mmr_lambda=LAMBDA)
    #
    # meta_arima4 = MetaARIMA(model=mod4,
    #                         freq=freq_str,
    #                         season_length=freq_int,
    #                         n_trials=N_TRIALS,
    #                         quantile_thr=QUANTILE_THR,
    #                         use_mmr=True,
    #                         base_optim=BASE_OPTIM,
    #                         mmr_lambda=LAMBDA)

    meta_arima1.meta_fit(X_train, y_train)
    meta_arima2.meta_fit(X_train, y_train)
    # meta_arima3.meta_fit(X_train, y_train)
    # meta_arima4.meta_fit(X_train, y_train)

    pred_list1 = meta_arima1.meta_predict(X_test)
    pred_list2 = meta_arima2.meta_predict(X_test)
    # pred_list3 = meta_arima3.meta_predict(X_test)
    # pred_list4 = meta_arima4.meta_predict(X_test)

    for i, (uid, x) in enumerate(X_test.iterrows()):
        print(i, uid)

        df_uid = train.query(f'unique_id=="{uid}"').copy()

        meta_arima1.fit(df_uid, config_space=pred_list1[i])
        meta_arima2.fit(df_uid, config_space=pred_list2[i])
        # meta_arima3.fit(df_uid, config_space=pred_list3[i])
        # meta_arima4.fit(df_uid, config_space=pred_list4[i])

        err_meta1 = cv.loc[uid, meta_arima1.selected_config]
        err_meta2 = cv.loc[uid, meta_arima2.selected_config]
        # err_meta3 = cv.loc[uid, meta_arima3.selected_config]
        # err_meta4 = cv.loc[uid, meta_arima4.selected_config]

        comp = {
            'MetaARIMA1': err_meta1,
            'MetaARIMA2': err_meta2,
            # 'MetaARIMA3': err_meta3,
            # 'MetaARIMA4': err_meta4,
        }

        pprint(comp)

        results.append(comp)

results_df = pd.DataFrame(results)
# results_df.to_csv(f'assets/results/sensitivity/algos,{data_name},{group}.csv', index=False)

print(results_df.mean())
print(results_df.median())
print(results_df.rank(axis=1).mean())
print(results_df.dropna().mean())
print(results_df.dropna().median())
print(results_df.dropna().rank(axis=1).mean())
