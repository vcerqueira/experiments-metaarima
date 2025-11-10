import numpy as np

ORDER_MAX = {'AR': 4, 'I': 1, 'MA': 4, 'S_AR': 1, 'S_I': 1, 'S_MA': 1}
ORDER_MAX_NONSEASONAL = {'AR': 4, 'I': 1, 'MA': 4, 'S_AR': 0, 'S_I': 0, 'S_MA': 0}
N_TRIALS_SPACE = [1, 2, 5, 10, 20, 30, 40, 50, 75, 94]
QUANTILE_THR = 0.5
PCA_N_COMPONENTS = 100
PCA_N_COMPONENTS_SPACE = [3, 10, 25, 50, 100, 150, 200, 250, 300, 350]
RANDOM_SEED = 1
LAMBDA = 0.75
BASE_OPTIM = 'halving'
MMR = True
N_TRIALS = 25
MAX_N_TRIALS = 94
N_LAMBDAS = 11
LAMBDA_SPACE = np.linspace(start=0, stop=1, num=N_LAMBDAS)
QUANTILE_SPACE = np.linspace(start=0.025, stop=.95, num=38)
N_FOLDS = 5

BEST_CATBOOST_PARAMS = {
    'm4_monthly': {'bootstrap_type': 'Bernoulli',
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
                   'verbose': False},
    'm4_quarterly': {'bootstrap_type': 'Bernoulli',
                     'border_count': 32,
                     'depth': 4,
                     'eval_metric': 'MultiRMSE',
                     'iterations': 266,
                     'l2_leaf_reg': 6.032217388079633,
                     'leaf_estimation_iterations': 2,
                     'learning_rate': 0.05961229266653637,
                     'loss_function': 'MultiRMSE',
                     'model_size_reg': 0.6153682157721689,
                     'od_type': 'Iter',
                     'od_wait': 20,
                     'random_seed': 42,
                     'rsm': 0.6992272121986959,
                     'task_type': 'CPU',
                     'use_best_model': False,
                     'verbose': False},

    'm4_yearly': {'bootstrap_type': 'Bernoulli',
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
}

BEST_LGBM_PARAMS = {
    'm4_monthly': {'colsample_bytree': 0.954674147279825,
                   'learning_rate': 0.08217295211648731,
                   'max_depth': 4,
                   'min_child_samples': 23,
                   'n_estimators': 970,
                   'num_leaves': 46,
                   'subsample': 0.8568198488145982}
}

BEST_RF_PARAMS = {
    'm4_monthly': {'colsample_bynode': 0.8018423785145038,
                   'learning_rate': 0.8121792213475358,
                   'n_estimators': 159,
                   'reg_lambda': 0.0007309399242521293,
                   'subsample': 0.8772547126671503}
}
