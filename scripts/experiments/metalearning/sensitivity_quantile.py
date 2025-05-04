import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._base import MetaARIMAUtils
from src.load_data.config import DATASETS
from src.config import (ORDER_MAX,
                        QUANTILE_SPACE,
                        LAMBDA,
                        N_TRIALS,
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

train, _ = data_loader.train_test_split(df, horizon=horizon)

cv = pd.read_csv(f'assets/metadata_cv/arima,{data_name},{group}.csv')
feats = pd.read_csv(f'assets/features/features,{data_name},{group}.csv')
input_variables = feats.set_index('unique_id').columns.tolist()

cv = cv.merge(feats, on=['unique_id']).set_index('unique_id')
# cv = cv.head(200)
# LAMBDA_SPACE = [.1, .2]

model_names = MetaARIMAUtils.get_models_sf(season_length=freq_int,
                                           return_names=True,
                                           max_config=ORDER_MAX)

X = cv.loc[:, input_variables].fillna(-1)
y = cv.loc[:, model_names]

quantile_results = {}
for quantile_ in QUANTILE_SPACE:
    print('QUANTILE', quantile_)

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

        meta_arima = MetaARIMA(model=mod,
                               freq=freq_str,
                               season_length=freq_int,
                               n_trials=N_TRIALS,
                               base_optim=BASE_OPTIM,
                               quantile_thr=quantile_,
                               use_mmr=MMR,
                               mmr_lambda=LAMBDA)

        meta_arima.meta_fit(X_train, y_train)

        print('MetaARIMA inference')
        pred_list = meta_arima.meta_predict(X_test)

        print('MetaARIMA evaluating')
        for i, (uid, x) in enumerate(X_test.iterrows()):
            print(i, uid)
            df_uid = train.query(f'unique_id=="{uid}"')

            try:
                meta_arima.fit(df_uid, config_space=pred_list[i])
            except ValueError:
                continue

            err_metaarima = cv.loc[uid, meta_arima.selected_config]

            results.append(err_metaarima)

    quantile_results[f'MetaARIMA({quantile_})'] = np.mean(results)

results_df = pd.Series(quantile_results)

print(results_df)
