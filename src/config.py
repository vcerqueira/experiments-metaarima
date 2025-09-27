import numpy as np

ORDER_MAX = {'AR': 4, 'I': 1, 'MA': 4, 'S_AR': 1, 'S_I': 1, 'S_MA': 1}
N_TRIALS_SPACE = [1, 2, 5, 10, 20, 30, 40, 50, 75, 94]
# QUANTILE_THR = 0.1
QUANTILE_THR = 0.25
RANDOM_SEED = 1
LAMBDA = 0.75
BASE_OPTIM = 'halving'
MMR = True
# N_TRIALS = 20
N_TRIALS = 25
MAX_N_TRIALS = 94
N_LAMBDAS = 11
LAMBDA_SPACE = np.linspace(start=0, stop=1, num=N_LAMBDAS)
QUANTILE_SPACE = np.linspace(start=0.025, stop=.5, num=20)
N_FOLDS = 5
