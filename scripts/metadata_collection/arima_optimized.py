import os
import sys
import logging

import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive, AutoTheta, AutoETS
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate
from src.load_data.config import DATASETS
from src.arima.meta import MetaARIMAUtils

# Ensure SciPy, NumPy, and OpenMP do not spawn unnecessary threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["NIXTLA_ID_AS_COL"] = "1"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_series(
    uid, uid_df, test, freq_str, freq_int, horizon, models, data_name, group
):
    """Function to process a single time series"""
    logging.info(f"Processing series: {uid}")

    sf_auto = StatsForecast(
        models=[
            AutoARIMA(),
            SeasonalNaive(season_length=freq_int),
            AutoETS(season_length=freq_int),
            AutoTheta(),
        ],
        freq=freq_str,
    )
    sf_auto.fit(df=uid_df)
    fcst_auto = sf_auto.predict(h=horizon)
    logging.info(f"AutoARIMA forecast completed for series: {uid}")

    arima_config = MetaARIMAUtils.get_model_order(
        sf_auto.fitted_[0][0].model_, as_alias=True, alias_freq=freq_int
    )

    sf = StatsForecast(models=models, freq=freq_str)
    try:
        sf.fit(df=uid_df)
    except ValueError:
        logging.error(f"ValueError encountered for series: {uid}")
        return None

    fcst = sf.predict(h=horizon)
    fcst = fcst.merge(test, on=["unique_id", "ds"], how="left")
    fcst_auto = fcst_auto.merge(test, on=["unique_id", "ds"], how="left")
    fcst = fcst.fillna(-1)
    logging.info(f"Forecast completed for series: {uid}")

    err = evaluate(df=fcst, metrics=[smape]).mean(numeric_only=True)
    err_auto = evaluate(df=fcst_auto, metrics=[smape]).mean(numeric_only=True)
    err_auto_ = {
        "score_AutoARIMA": err_auto["AutoARIMA"],
        "score_SeasNaive": err_auto["SeasonalNaive"],
        "score_AutoTheta": err_auto["AutoTheta"],
        "score_AutoETS": err_auto["AutoETS"],
    }

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
            **err_auto_,
            "best_config": best_model_name,
            "auto_config": arima_config,
            "dataset": f"{data_name},{group}",
            "unique_id": uid,
        },
    }


if __name__ == "__main__":  # REQUIRED for multiprocessing
    # data_name, group = 'M3', 'Monthly'
    # data_name, group = 'M3', 'Quarterly'
    # data_name, group = "Tourism", "Monthly"
    # data_name, group = "Tourism", "Quarterly"
    data_name, group = "M4", "Monthly"
    # data_name, group = "M4", "Quarterly"
    # data_name, group = "M4", "Weekly"
    # data_name, group = "M4", "Daily"

    logging.info(f"Starting processing for dataset: {data_name}, group: {group}")

    data_loader = DATASETS[data_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, extended=True
    )
    logging.info(f"Data loaded with horizon: {horizon}, frequency: {freq_str}")

    train, test = data_loader.train_test_split(df, horizon=horizon)
    logging.info(f"Train and test data split with horizon: {horizon}")

    ORDER_MAX = {"AR": 4, "I": 1, "MA": 4, "S_AR": 1, "S_I": 1, "S_MA": 1}
    models = MetaARIMAUtils.get_models_sf(season_length=freq_int, max_config=ORDER_MAX)
    logging.info(f"Number of models: {len(models)}")

    PREV_RESULTS_CSV = [f"arima,{data_name},{group}_.csv"]
    outfile = Path(__file__).resolve().parents[2] / "assets" / "metadata_cv"

    result_files = set()
    for file in PREV_RESULTS_CSV:
        with open(outfile / file, "r") as f:
            r = pd.read_csv(f)
            result_files.update(r["unique_id"].values.tolist())
    logging.info(f"Previous result files loaded: {len(result_files)}")

    df_grouped = train.groupby("unique_id")
    results = {}

    with ProcessPoolExecutor() as executor:
        future_to_uid = {}
        for uid, uid_df in df_grouped:
            if uid in result_files:
                logging.info(f"Skipping unique_id: {uid} as it is already processed")
                continue
            future_to_uid[
                executor.submit(
                    process_series,
                    uid,
                    uid_df,
                    test,
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
    results_df.to_csv(outfile / f"arima,{data_name},{group}.csv", index=False)
    logging.info("Processing completed and results saved")
