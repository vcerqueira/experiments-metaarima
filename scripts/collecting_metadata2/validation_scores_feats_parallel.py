from pprint import pprint
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
import logging
import warnings

warnings.filterwarnings("ignore")

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from statsforecast.models import AutoARIMA, SeasonalNaive, AutoTheta, AutoETS
from statsforecast import StatsForecast
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate

from src.meta.arima._base import MetaARIMAUtils
from src.config import ORDER_MAX
from src.load_data.config import DATASETS

# data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
data_name, group = 'M4', 'Monthly'
# data_name, group = 'M4', 'Quarterly'
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=3)

train_all, _ = data_loader.train_test_split(df, horizon=horizon)
dev, validation = data_loader.train_test_split(train_all, horizon=horizon)

models = MetaARIMAUtils.get_models_sf(season_length=freq_int, max_config=ORDER_MAX)
print(len(models))

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
        uid, uid_df, test, freq_str, freq_int, horizon, models, data_name, group
):
    """Function to process a single time series"""
    logging.info(f"Processing series: {uid}")

    # sf_auto = StatsForecast(
    #     models=[
    #         AutoARIMA(),
    #         SeasonalNaive(season_length=freq_int),
    #         AutoETS(season_length=freq_int),
    #         AutoTheta(),
    #     ],
    #     freq=freq_str,
    # )
    # sf_auto.fit(df=uid_df)
    # fcst_auto = sf_auto.predict(h=horizon)
    # logging.info(f"AutoARIMA forecast completed for series: {uid}")

    # arima_config = MetaARIMAUtils.get_model_order(
    #     sf_auto.fitted_[0][0].model_, as_alias=True, alias_freq=freq_int
    # )

    sf = StatsForecast(models=models, freq=freq_str)
    try:
        sf.fit(df=uid_df)
    except ValueError:
        logging.error(f"ValueError encountered for series: {uid}")
        return None

    fcst = sf.predict(h=horizon)
    fcst = fcst.merge(test, on=["unique_id", "ds"], how="left")
    # fcst_auto = fcst_auto.merge(test, on=["unique_id", "ds"], how="left")
    fcst = fcst.fillna(-1)
    logging.info(f"Forecast completed for series: {uid}")

    err = evaluate(df=fcst, metrics=[smape]).mean(numeric_only=True)

    # we don't need this info during validation
    # err_auto = evaluate(df=fcst_auto, metrics=[smape]).mean(numeric_only=True)
    # err_auto_ = {
    #     "score_AutoARIMA": err_auto["AutoARIMA"],
    #     "score_SeasNaive": err_auto["SeasonalNaive"],
    #     "score_AutoTheta": err_auto["AutoTheta"],
    #     "score_AutoETS": err_auto["AutoETS"],
    # }

    best_model_name = err.sort_values().index[0]
    best_model = sf.fitted_.flatten()[err.argmin()]
    mod_summary = MetaARIMAUtils.model_summary(best_model.model_)
    logging.info(f"Model summary for best model of series {uid}: {mod_summary}")

    logging.info(f"Finished processing series: {uid}")

    return {
        "uid": uid,
        "results": {
            **err.to_dict(),
            **mod_summary,
            'best_config': best_model_name,
            'dataset': f'{data_name},{group}',
            'unique_id': uid,
        },
    }


if __name__ == "__main__":
    outfile_arima = Path(__file__).parent.parent.parent / 'assets' / 'metadata_cv'

    PREV_RESULTS_CSV = ["arima-dev,M4,Monthly.csv"]

    result_files = []
    for file in PREV_RESULTS_CSV:
        file_path = outfile_arima / file
        if file_path.exists():
            r = pd.read_csv(file_path)
            result_files += r["unique_id"].values.tolist()

    dev_missing = dev[~dev["unique_id"].isin(result_files)]

    df_grouped = dev_missing.groupby("unique_id")

    results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_uid = {}
        for uid, uid_df in df_grouped:
            # if uid in result_files:
            #     logging.info(f"Skipping unique_id: {uid} as it is already processed")
            #     continue

            future_to_uid[
                executor.submit(
                    process_series,
                    uid,
                    uid_df,
                    validation,
                    freq_str,
                    freq_int,
                    horizon,
                    models,
                    data_name,
                    group,
                )
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
        if col not in ["best_config", "dataset", "auto_config", "unique_id"]
    }
    results_df = results_df.astype(type_dict)
    results_df.to_csv(outfile_arima / f"arima-dev,{data_name},{group},2.csv", index=False)
    logging.info("Processing completed and results saved")
