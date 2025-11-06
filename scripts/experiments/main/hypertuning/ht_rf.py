from pprint import pprint

from xgboost import XGBRFRegressor
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
    'n_estimators': randint(25, 300),
    'learning_rate': uniform(0.1, 1),
    'subsample': uniform(0.6, 0.9),
    'colsample_bynode': uniform(0.6, 0.9),
    'reg_lambda': uniform(0.00001, 0.001),
}

base_model = XGBRFRegressor()

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

random_search.fit(X, y_pca)

print("Best parameters:", random_search.best_params_)
print("Best validation score:", random_search.best_score_)
pprint(random_search.best_params_)

BEST_RF_PARAMS = {}
