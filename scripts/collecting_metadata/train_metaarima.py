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

# model = CatBoostRegressor(
#     loss_function="MultiRMSE",
#     eval_metric="MultiRMSE",
#     iterations=300,
#     depth=3,
#     learning_rate=0.05,
#     random_seed=42,
# )

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

BEST_LGBM_PARAMS = {'colsample_bytree': 0.8468358280832688,
                    'learning_rate': 0.10856504541106007,
                    'max_depth': 4,
                    'min_child_samples': 77,
                    'n_estimators': 282,
                    'num_leaves': 142,
                    'subsample': 0.7726479814832277}

# model = MultiOutputRegressor(LGBMRegressor(**BEST_LGBM_PARAMS))
model = CatBoostRegressor(**BEST_CATBOOST_PARAMS)
# model = XGBRFRegressor()

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
