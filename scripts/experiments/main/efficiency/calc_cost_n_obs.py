from sklearn.model_selection import KFold
from xgboost import XGBRFRegressor

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.load_data.config import DATASETS
from src.config import MMR, N_TRIALS, QUANTILE_THR, BASE_OPTIM, LAMBDA

data_name, group = 'M3', 'Monthly'
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, extended=True)

train, _ = data_loader.train_test_split(df, horizon=horizon)

mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=freq_int)

X_dev, y_dev, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1)
X, _, _, _, cv_test = mdr.read(from_dev_set=False, fill_na_value=-1)
print(cv_test.shape)

# note that this is cv on the time series set (80% of time series for train, 20% for testing)
# partition is done at time series level, not in time dimension
kfcv = KFold(n_splits=5, random_state=1, shuffle=True)

tot_obs_metaarima, tot_obs_autoarima = 0, 0
# j, (train_index, test_index) = next(enumerate(kfcv.split(X)))
for j, (train_index, test_index) in enumerate(kfcv.split(X)):
    print(f"Fold {j}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

    X_train = X_dev.iloc[train_index, :]
    y_train = y_dev.iloc[train_index, :]
    X_test = X.iloc[test_index, :]

    meta_arima = MetaARIMA(model=XGBRFRegressor(),
                           freq=freq_str,
                           season_length=freq_int,
                           n_trials=N_TRIALS,
                           quantile_thr=QUANTILE_THR,
                           use_mmr=MMR,
                           base_optim=BASE_OPTIM,
                           mmr_lambda=LAMBDA)

    meta_arima.meta_fit(X_train, y_train)

    pred_list = meta_arima.meta_predict(X_test)

    for i, uid in enumerate(X_test.index):
        print(i, uid)
        # if i > 3:
        #     break

        df_uid = train.query(f'unique_id=="{uid}"').copy()
        # default is 94 models
        print('shape:', df_uid.shape[0])
        tot_obs_autoarima += df_uid.shape[0] * 94

        meta_arima.fit(df_uid, config_space=pred_list[i])
        print('n obs:', meta_arima.model.tot_nobs)

        tot_obs_metaarima += meta_arima.model.tot_nobs
        tot_obs_metaarima += df_uid.shape[0]  # for the feature extraction step

print('metaarima tot:', tot_obs_metaarima)
print('autoarima tot:', tot_obs_autoarima)
# metaarima tot: 4847310
# autoarima tot: 14140044
