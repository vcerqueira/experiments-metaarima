from pprint import pprint

from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.decomposition import PCA

from src.meta.arima._data_reader import MetadataReader
from src.chronos_data import ChronosDataset
from src.config import QUANTILE_THR, PCA_N_COMPONENTS

source = 'm4_monthly'
_, _, _, freq_str, freq_int = ChronosDataset.load_everything(source)

mdr = MetadataReader(group=source, freq_int=freq_int)
X, y, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1)

y = y.apply(lambda x: (x <= x.quantile(QUANTILE_THR)).astype(int), axis=1)
pca = PCA(n_components=PCA_N_COMPONENTS)
y_pca = pca.fit_transform(y)

param_distributions = {
    'n_estimators': randint(100, 600),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(2, 5),
    'num_leaves': randint(20, 150),
    'min_child_samples': randint(10, 100),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
}

base_model = LGBMRegressor(verbosity=-1)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=6,
    scoring='neg_mean_absolute_error',
    n_jobs=3,
    random_state=42,
    verbose=2
)

random_search.fit(X, y_pca[:, 0])

print("Best parameters:", random_search.best_params_)
print("Best validation score:", random_search.best_score_)
pprint(random_search.best_params_)

BEST_LGBM_PARAMS = {'colsample_bytree': 0.8468358280832688,
                    'learning_rate': 0.10856504541106007,
                    'max_depth': 4,
                    'min_child_samples': 77,
                    'n_estimators': 282,
                    'num_leaves': 142,
                    'subsample': 0.7726479814832277}

