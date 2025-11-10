from catboost import CatBoostRegressor

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
                        ORDER_MAX_NONSEASONAL,
                        BEST_CATBOOST_PARAMS)

# -- train metamodel
algorithm = 'catboost'
source = 'm4_monthly'
FILENAME = f'assets/trained_metaarima_{source}_{algorithm}.joblib.gz'
_, _, _, freq_str, freq_int = ChronosDataset.load_everything(source)

ord = ORDER_MAX if freq_int > 1 else ORDER_MAX_NONSEASONAL
pca_n = PCA_N_COMPONENTS if freq_int > 1 else 10

mdr = MetadataReader(group=source, freq_int=freq_int)
X, y, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1, max_config=ord)

model = CatBoostRegressor(**BEST_CATBOOST_PARAMS[source])

meta_arima = MetaARIMA(model=model,
                       freq=freq_str,
                       season_length=freq_int,
                       n_trials=N_TRIALS,
                       quantile_thr=QUANTILE_THR,
                       pca_n_components=pca_n,
                       use_mmr=MMR,
                       base_optim=BASE_OPTIM,
                       mmr_lambda=LAMBDA)

meta_arima.meta_fit(X, y)

ModelIO.save_model(meta_arima, FILENAME)
