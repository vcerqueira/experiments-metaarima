# ideally this should compare metaarima@N with autoarima@N, not default autoarima
from pprint import pprint
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

train, test = data_loader.train_test_split(df, horizon=horizon)

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


# maybe just iterate over the quantile space and get the final result
## todo just create lots of metaarima's

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

    scores = {}
    for quantile_ in QUANTILE_SPACE:

        print('MetaARIMA fitting')
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
                scores[f'MetaARIMA({quantile_})'] = np.nan
                continue

            mod_ = meta_arima.model.sf.fitted_[0][0]
            config_selected_mrr = MetaARIMAUtils.get_model_order(mod_.model_,
                                                                 as_alias=True,
                                                                 alias_freq=freq_int)

            auto_arima_config = cv.loc[uid, 'auto_config']
            err_meta_mmr = cv.loc[uid, config_selected_mrr]

            scores[f'MetaARIMA({quantile_})'] = err_meta_mmr
            scores['AutoARIMA'] = cv.loc[uid, 'score_AutoARIMA']
            scores['unique_id'] = f'{data_name},{group},{uid}'

            pprint(scores)

            results.append(scores)

results_df = pd.DataFrame(results)

print(results_df.mean())
print(results_df.median())
print(results_df.rank(axis=1, na_option='bottom').mean())
print(results_df.dropna().mean())
print(results_df.dropna().median())
print(results_df.dropna().rank(axis=1).mean())
