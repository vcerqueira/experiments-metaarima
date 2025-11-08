from xgboost import XGBRFRegressor, XGBRFClassifier
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader, ModelIO
from src.chronos_data import ChronosDataset
from src.config import (MMR,
                        N_TRIALS,
                        QUANTILE_THR,
                        BASE_OPTIM,
                        LAMBDA,
                        PCA_N_COMPONENTS,
                        ORDER_MAX,
                        ORDER_MAX_NONSEASONAL)

# -- train metamodel
algorithm = 'catboost'
source = 'm4_yearly'
FILENAME = f'assets/trained_metaarima_{source}_{algorithm}.joblib.gz'
_, _, _, freq_str, freq_int = ChronosDataset.load_everything(source)

ord = ORDER_MAX if freq_int > 1 else ORDER_MAX_NONSEASONAL
pca_n = PCA_N_COMPONENTS if freq_int > 1 else 10

mdr = MetadataReader(group=source, freq_int=freq_int)
X, y, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1, max_config=ord)

# BEST_CATBOOST_PARAMS = {'bootstrap_type': 'Bernoulli',
#                         'border_count': 32,
#                         'depth': 4,
#                         'eval_metric': 'MultiRMSE',
#                         'iterations': 283,
#                         'l2_leaf_reg': 16.732679992180078,
#                         'leaf_estimation_iterations': 2,
#                         'learning_rate': 0.04248616344796388,
#                         'loss_function': 'MultiRMSE',
#                         'model_size_reg': 2.0916597401644705,
#                         'od_type': 'Iter',
#                         'od_wait': 60,
#                         'random_seed': 42,
#                         'rsm': 0.6406208143252348,
#                         'task_type': 'CPU',
#                         'use_best_model': False,
#                         'verbose': False}


# quarterly
# BEST_CATBOOST_PARAMS = {'bootstrap_type': 'Bernoulli',
#                         'border_count': 32,
#                         'depth': 4,
#                         'eval_metric': 'MultiRMSE',
#                         'iterations': 266,
#                         'l2_leaf_reg': 6.032217388079633,
#                         'leaf_estimation_iterations': 2,
#                         'learning_rate': 0.05961229266653637,
#                         'loss_function': 'MultiRMSE',
#                         'model_size_reg': 0.6153682157721689,
#                         'od_type': 'Iter',
#                         'od_wait': 20,
#                         'random_seed': 42,
#                         'rsm': 0.6992272121986959,
#                         'task_type': 'CPU',
#                         'use_best_model': False,
#                         'verbose': False}

# y
BEST_CATBOOST_PARAMS = {'bootstrap_type': 'Bernoulli',
                        'border_count': 64,
                        'depth': 4,
                        'eval_metric': 'MultiRMSE',
                        'iterations': 300,
                        'l2_leaf_reg': 5.903373826675397,
                        'leaf_estimation_iterations': 2,
                        'learning_rate': 0.05544986984539306,
                        'loss_function': 'MultiRMSE',
                        'model_size_reg': 1.5546708502742612,
                        'od_type': 'Iter',
                        'od_wait': 70,
                        'random_seed': 42,
                        'rsm': 0.8381786985397336,
                        'task_type': 'CPU',
                        'use_best_model': False,
                        'verbose': False}

BEST_LGBM_PARAMS = {'colsample_bytree': 0.954674147279825,
                    'learning_rate': 0.08217295211648731,
                    'max_depth': 4,
                    'min_child_samples': 23,
                    'n_estimators': 970,
                    'num_leaves': 46,
                    'subsample': 0.8568198488145982}

BEST_RF_PARAMS = {'colsample_bynode': 0.8018423785145038,
                  'learning_rate': 0.8121792213475358,
                  'n_estimators': 159,
                  'reg_lambda': 0.0007309399242521293,
                  'subsample': 0.8772547126671503}

# model = MultiOutputRegressor(LGBMRegressor(**BEST_LGBM_PARAMS))
model = CatBoostRegressor(**BEST_CATBOOST_PARAMS)
# model = XGBRFClassifier()
# model = XGBRFRegressor()
# model = XGBRFRegressor(**BEST_RF_PARAMS)

meta_arima = MetaARIMA(model=model,
                       freq=freq_str,
                       season_length=freq_int,
                       n_trials=N_TRIALS,
                       # n_trials=int(N_TRIALS/2),
                       quantile_thr=QUANTILE_THR,
                       # quantile_thr=.2,
                       # meta_regression=True,
                       # target_pca=False,
                       pca_n_components=pca_n,
                       use_mmr=MMR,
                       base_optim=BASE_OPTIM,
                       mmr_lambda=LAMBDA)

meta_arima.meta_fit(X, y)

ModelIO.save_model(meta_arima, FILENAME)
