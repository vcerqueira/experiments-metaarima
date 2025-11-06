import os
import logging
import warnings
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import pandas as pd
from statsforecast import StatsForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate

from src.meta.arima._base import MetaARIMAUtils
from src.config import ORDER_MAX, ORDER_MAX_NONSEASONAL
from src.chronos_data import ChronosDataset

warnings.filterwarnings("ignore")

GROUP = 'm4_monthly'
# GROUP = 'm4_quarterly'
# GROUP = 'm4_yearly'

df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(GROUP)

# discard test
train, _ = ChronosDataset.time_wise_split(df, horizon=horizon)
# split train into dev and validation
dev, validation = ChronosDataset.time_wise_split(train, horizon=horizon)

ord = ORDER_MAX if seas_len > 1 else ORDER_MAX_NONSEASONAL

models = MetaARIMAUtils.get_models_sf(season_length=seas_len, max_config=ord)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NIXTLA_ID_AS_COL"] = "1"

MAX_WORKERS = 8

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_series(
        uid: str,
        uid_insample: pd.DataFrame,
        outsample: pd.DataFrame,
        freq_str: str,
        seas_len: int,
        horizon: int,
        models: List):
    """Function to process a single time series"""
    logging.info(f"Processing series: {uid}")

    sf = StatsForecast(models=models, freq=freq_str)
    try:
        sf.fit(df=uid_insample)
    except ValueError:
        logging.error(f"ValueError encountered for series: {uid}")
        return None

    fcst = sf.predict(h=horizon)
    fcst = fcst.merge(outsample, on=["unique_id", "ds"], how='left')
    logging.info(f"Forecast completed for series: {uid}")

    err_df = evaluate(df=fcst,
                      # metrics=[partial(mase, seasonality=seas_len)],
                      metrics=[smape],
                      train_df=uid_insample)

    avg_err = err_df.mean(numeric_only=True)

    best_model_name = avg_err.sort_values().index[0]
    best_model = sf.fitted_.flatten()[avg_err.argmin()]
    mod_summary = MetaARIMAUtils.model_summary(best_model.model_)
    logging.info(f"Model summary for best model of series {uid}: {mod_summary}")

    logging.info(f"Finished processing series: {uid}")

    return {
        "uid": uid,
        "results": {
            **avg_err.to_dict(),
            **mod_summary,
            'best_config': best_model_name,
            'unique_id': uid,
        },
    }


if __name__ == "__main__":
    outfile_arima = Path(__file__).parent.parent.parent / 'assets' / 'metadata'

    df_grouped = dev.groupby("unique_id")

    results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_uid = {}
        for uid, uid_df in df_grouped:
            future_to_uid[
                executor.submit(
                    process_series,
                    uid=uid,
                    uid_insample=uid_df,
                    outsample=validation,
                    freq_str=freq,
                    seas_len=seas_len,
                    horizon=horizon,
                    models=models)
            ] = uid

        for future in as_completed(future_to_uid):
            uid = future_to_uid[future]
            try:
                result = future.result()
                if result:
                    results[result["uid"]] = result["results"]
                    logging.info(f"Result collected for series: {uid}")
            except Exception as e:
                logging.error(f"Error processing series {uid}: {e}")

    results_df = pd.DataFrame.from_dict(results, orient="index")
    type_dict = {
        col: float
        for col in results_df.columns
        if col not in ["best_config", "unique_id"]
    }
    results_df = results_df.astype(type_dict)
    results_df.to_csv(outfile_arima / f"arima,{GROUP}.csv", index=False)
    logging.info("Processing completed and results saved")
