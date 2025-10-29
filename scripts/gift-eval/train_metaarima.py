from xgboost import XGBRFRegressor

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader, ModelIO
from src.load_data.config import DATASETS
from src.config import (MMR,
                        N_TRIALS,
                        QUANTILE_THR,
                        BASE_OPTIM,
                        LAMBDA,
                        PCA_N_COMPONENTS)

FILENAME = 'assets/trained_metaarima_m4m.joblib.gz'
# -- train metamodel
source_data_name, source_group = 'M4', 'Monthly'
data_loader = DATASETS[source_data_name]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(source_group, extended=True)
train, _ = data_loader.train_test_split(df, horizon=horizon)
mdr = MetadataReader(dataset_name=source_data_name, group=source_group, freq_int=freq_int)
X, y, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1)

meta_arima = MetaARIMA(model=XGBRFRegressor(),
                       freq=freq_str,
                       season_length=freq_int,
                       n_trials=N_TRIALS,
                       quantile_thr=QUANTILE_THR,
                       # pca_n_components=PCA_N_COMPONENTS,
                       pca_n_components=50,  # for smaller model
                       use_mmr=MMR,
                       base_optim=BASE_OPTIM,
                       mmr_lambda=LAMBDA)

meta_arima.meta_fit(X, y)

ModelIO.save_model(meta_arima, FILENAME)


loaded_metaarima = ModelIO.load_model(FILENAME)

preds = loaded_metaarima.meta_predict(X.head(2))

print(preds)

example_df = df.query('unique_id=="M1"')
freq_int = 12


feat_df = tsfeatures_uid(example_df, freq_int)

config_space = loaded_metaarima.meta_predict(feat_df)[0]

loaded_metaarima.fit(example_df, config_space)
loaded_metaarima.model.predict(h=12)