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
