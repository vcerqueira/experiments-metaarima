import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRFRegressor

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.load_data.config import DATASETS
from src.config import (QUANTILE_THR,
                        LAMBDA,
                        N_TRIALS,
                        PCA_N_COMPONENTS_SPACE,
                        MMR,
                        BASE_OPTIM,
                        N_FOLDS,
                        RANDOM_SEED)

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

mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=freq_int)

X_dev, y_dev, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1)
X, _, _, _, cv_test = mdr.read(from_dev_set=False, fill_na_value=-1)
print(cv_test.shape)

ncomps_results = {}
for n_comps_ in PCA_N_COMPONENTS_SPACE:
    print('N COMPS', n_comps_)

    kfcv = KFold(n_splits=N_FOLDS, random_state=RANDOM_SEED, shuffle=True)

    results, explained_var_ = [], []
    for j, (train_index, test_index) in enumerate(kfcv.split(X)):
        print(f"Fold {j}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        X_train = X_dev.iloc[train_index, :]
        y_train = y_dev.iloc[train_index, :]
        X_test = X.iloc[test_index, :]

        mod = XGBRFRegressor()

        meta_arima = MetaARIMA(model=mod,
                               freq=freq_str,
                               season_length=freq_int,
                               n_trials=N_TRIALS,
                               base_optim=BASE_OPTIM,
                               pca_n_components=n_comps_,
                               quantile_thr=QUANTILE_THR,
                               use_mmr=MMR,
                               mmr_lambda=LAMBDA)

        meta_arima.meta_fit(X_train, y_train)

        exp_var = meta_arima.meta_model.pca.explained_variance_ratio_
        print(f'Explained var at {n_comps_}: {exp_var}')
        explained_var_.append(exp_var)

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

            err_metaarima = cv_test.loc[uid, meta_arima.selected_config]

            results.append(err_metaarima)

    ncomps_results[f'MetaARIMA({str(n_comps_)})'] = {'avg': np.mean(results),
                                                     'med': np.median(results),
                                                     'std': np.std(results),
                                                     'avg_expl_var': np.mean(explained_var_)
                                                     }

results_df = pd.DataFrame(ncomps_results).T
results_df.to_csv(f'assets/results/sensitivity/ncomps,{data_name},{group}.csv')

print(results_df)
