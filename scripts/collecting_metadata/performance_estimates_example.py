import logging
import warnings
from functools import partial

warnings.filterwarnings("ignore")

from statsforecast import StatsForecast
from utilsforecast.losses import mase
from utilsforecast.evaluation import evaluate

from src.meta.arima._base import MetaARIMAUtils
from src.config import ORDER_MAX
from src.chronos_data import ChronosDataset

GROUP = 'monash_m1_monthly'

# df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(GROUP)
df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(GROUP, sample_n_uid=1)
df = ChronosDataset.dummify_series(df)

# discard test
train, _ = ChronosDataset.time_wise_split(df, horizon=horizon)
# split train into dev and validation
dev, validation = ChronosDataset.time_wise_split(train, horizon=horizon)

models = MetaARIMAUtils.get_models_sf(season_length=seas_len, max_config=ORDER_MAX)
print(len(models))


sf = StatsForecast(models=models, freq=freq)
sf.fit(df=dev)

fcst = sf.predict(h=horizon)
fcst = fcst.merge(validation, on=["unique_id", "ds"])

err_df = evaluate(df=fcst,
                  metrics=[partial(mase, seasonality=seas_len)],
                  train_df=dev)
avg_err = err_df.mean(numeric_only=True)

best_model_name = avg_err.sort_values().index[0]
best_model = sf.fitted_.flatten()[avg_err.argmin()]
mod_summary = MetaARIMAUtils.model_summary(best_model.model_)


