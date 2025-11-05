from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor

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


model = CatBoostRegressor(
    loss_function="MultiRMSE",
    eval_metric="MultiRMSE",
    iterations=300,
    depth=8,
    learning_rate=0.05,
    random_seed=42,
)

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
