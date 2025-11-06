from xgboost import XGBRFRegressor
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
                        PCA_N_COMPONENTS)

FILENAME = 'assets/trained_metaarima_m4m_cb.joblib.gz'
# -- train metamodel
source = 'm4_monthly'
_, _, _, freq_str, freq_int = ChronosDataset.load_everything(source)

mdr = MetadataReader(group=source, freq_int=freq_int)
X, y, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1)

BEST_CATBOOST_PARAMS = {'bootstrap_type': 'Bernoulli',
                        'border_count': 32,
                        'depth': 4,
                        'eval_metric': 'MultiRMSE',
                        'iterations': 283,
                        'l2_leaf_reg': 16.732679992180078,
                        'leaf_estimation_iterations': 2,
                        'learning_rate': 0.04248616344796388,
                        'loss_function': 'MultiRMSE',
                        'model_size_reg': 2.0916597401644705,
                        'od_type': 'Iter',
                        'od_wait': 60,
                        'random_seed': 42,
                        'rsm': 0.6406208143252348,
                        'task_type': 'CPU',
                        'use_best_model': False,
                        'verbose': False}

#
# BEST_CATBOOST_PARAMS = {'bootstrap_type': 'Bernoulli',
#                         'border_count': 64,
#                         'depth': 4,
#                         'eval_metric': 'MultiRMSE',
#                         'iterations': 1000,
#                         'l2_leaf_reg': 44.766640345162486,
#                         'leaf_estimation_iterations': 3,
#                         'learning_rate': 0.05542045515461307,
#                         'loss_function': 'MultiRMSE',
#                         'model_size_reg': 2.490479898592805,
#                         'od_type': 'Iter',
#                         'od_wait': 50,
#                         'random_seed': 42,
#                         'rsm': 0.8036450022637952,
#                         'task_type': 'CPU',
#                         'use_best_model': False,
#                         'verbose': False}

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
# model = XGBRFRegressor()
# model = XGBRFRegressor(**BEST_RF_PARAMS)

meta_arima = MetaARIMA(model=model,
                       freq=freq_str,
                       season_length=freq_int,
                       n_trials=N_TRIALS,
                       quantile_thr=QUANTILE_THR,
                       pca_n_components=PCA_N_COMPONENTS,
                       use_mmr=MMR,
                       base_optim=BASE_OPTIM,
                       mmr_lambda=LAMBDA)

meta_arima.meta_fit(X, y)

ModelIO.save_model(meta_arima, FILENAME)
