from pprint import pprint

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from lightgbm import LGBMClassifier

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._base import MetaARIMAUtils
from src.load_data.config import DATASETS
from src.config import N_TRIALS, QUANTILE_THR, BASE_OPTIM, LAMBDA

data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
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

    mod = ClassifierChain(LGBMClassifier(verbosity=-1))
    mod_mo = MultiOutputClassifier(LGBMClassifier(verbosity=-1))

    meta_arima = MetaARIMA(model=mod,
                           freq=freq_str,
                           season_length=freq_int,
                           n_trials=N_TRIALS,
                           quantile_thr=QUANTILE_THR,
                           use_mmr=True,
                           base_optim=BASE_OPTIM,
                           mmr_lambda=LAMBDA)

    meta_arima_nosh = MetaARIMA(model=mod,
                                freq=freq_str,
                                season_length=freq_int,
                                n_trials=N_TRIALS,
                                quantile_thr=QUANTILE_THR,
                                use_mmr=True,
                                base_optim='complete',
                                mmr_lambda=LAMBDA)

    meta_arima_mc = MetaARIMA(model=mod,
                                freq=freq_str,
                                season_length=freq_int,
                                n_trials=N_TRIALS,
                                quantile_thr=QUANTILE_THR,
                                use_mmr=True,
                                base_optim='mc',
                                mmr_lambda=LAMBDA)

    meta_arima_mo = MetaARIMA(model=mod_mo,
                              freq=freq_str,
                              season_length=freq_int,
                              n_trials=N_TRIALS,
                              quantile_thr=QUANTILE_THR,
                              use_mmr=True,
                              base_optim=BASE_OPTIM,
                              mmr_lambda=LAMBDA)

    meta_arima_nommr = MetaARIMA(model=mod,
                                 freq=freq_str,
                                 season_length=freq_int,
                                 n_trials=N_TRIALS,
                                 quantile_thr=QUANTILE_THR,
                                 use_mmr=False,
                                 base_optim=BASE_OPTIM,
                                 mmr_lambda=LAMBDA)

    meta_arima.meta_fit(X_train, y_train)
    meta_arima_nosh.meta_fit(X_train, y_train)
    meta_arima_mc.meta_fit(X_train, y_train)
    meta_arima_mo.meta_fit(X_train, y_train)
    meta_arima_nommr.meta_fit(X_train, y_train)

    pred_list = meta_arima.meta_predict(X_test)
    pred_list_nosh = meta_arima_nosh.meta_predict(X_test)
    pred_list_mc = meta_arima_mc.meta_predict(X_test)
    pred_list_mo = meta_arima_mo.meta_predict(X_test)
    pred_list_nommr = meta_arima_nommr.meta_predict(X_test)

    for i, (uid, x) in enumerate(X_test.iterrows()):
        print(i, uid)

        df_uid = train.query(f'unique_id=="{uid}"').copy()

        meta_arima.fit(df_uid, config_space=pred_list[i])
        meta_arima_nosh.fit(df_uid, config_space=pred_list_nosh[i])
        meta_arima_mc.fit(df_uid, config_space=pred_list_mc[i])
        meta_arima_mo.fit(df_uid, config_space=pred_list_mo[i])
        meta_arima_nommr.fit(df_uid, config_space=pred_list_nommr[i])

        err_meta = cv.loc[uid, meta_arima.selected_config]
        err_meta_nosh = cv.loc[uid, meta_arima_nosh.selected_config]
        err_meta_mc = cv.loc[uid, meta_arima_mc.selected_config]
        err_meta_mo = cv.loc[uid, meta_arima_mo.selected_config]
        err_meta_nommr = cv.loc[uid, meta_arima_nommr.selected_config]

        comp = {
            'MetaARIMA': err_meta,
            'MetaARIMA(No-SH)': err_meta_nosh,
            'MetaARIMA(MC)': err_meta_mc,
            'MetaARIMA(MO)': err_meta_mo,
            'MetaARIMA(No-MMR)': err_meta_nommr,
        }

        pprint(comp)

        results.append(comp)

results_df = pd.DataFrame(results)
results_df.to_csv(f'assets/results/sensitivity/ablation,{data_name},{group}.csv', index=False)

print(results_df.mean())
print(results_df.median())
print(results_df.rank(axis=1).mean())
print(results_df.dropna().mean())
print(results_df.dropna().median())
print(results_df.dropna().rank(axis=1).mean())
