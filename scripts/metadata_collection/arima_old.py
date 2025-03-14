import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

os.environ["NIXTLA_ID_AS_COL"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pprint import pprint
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive, AutoTheta, AutoETS
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate

from src.load_data.config import DATASETS
from src.arima.meta import MetaARIMAUtils

# data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
data_name, group = "M4", "Monthly"
# data_name, group = 'M4', 'Quarterly'

logging.info(f"Data name: {data_name}, Group: {group}")
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
logging.info(f"Data loaded with horizon: {horizon}, frequency: {freq_str}")

train, test = data_loader.train_test_split(df, horizon=horizon)
logging.info(f"Train and test data split with horizon: {horizon}")

ORDER_MAX = {"AR": 4, "I": 1, "MA": 4, "S_AR": 1, "S_I": 1, "S_MA": 1}
models = MetaARIMAUtils.get_models_sf(season_length=freq_int, max_config=ORDER_MAX)
logging.info(f"Number of models: {len(models)}")

PREV_RESULTS_CSV = ["arima,M4,Monthly_.csv"]

if __name__ == "__main__":
    outfile = Path(__file__).resolve().parents[2] / "assets" / "metadata_cv"

    result_files = []
    for file in PREV_RESULTS_CSV:
        r = pd.read_csv(outfile / file)
        result_files += r["unique_id"].values.tolist()
    logging.info(f"Previous result files loaded: {len(result_files)}")

    results = {}
    df_grouped = train.groupby("unique_id")
    for uid, uid_df in df_grouped:
        logging.info(f"Processing unique_id: {uid}")
        if uid in result_files:
            logging.info(f"Skipping unique_id: {uid} as it is already processed")
            continue

        # df = ds.query('unique_id=="Y1"')
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
        logging.info(f"AutoARIMA forecast completed for unique_id: {uid}")

        arima_config = MetaARIMAUtils.get_model_order(
            sf_auto.fitted_[0][0].model_, as_alias=True, alias_freq=freq_int
        )

        sf = StatsForecast(models=models, freq=freq_str)
        try:
            sf.fit(df=uid_df)
        except ValueError:
            logging.error(
                f"ValueError encountered while fitting models for unique_id: {uid}"
            )
            continue

        fcst = sf.predict(h=horizon)
        fcst = fcst.merge(test, on=["unique_id", "ds"], how="left")
        fcst_auto = fcst_auto.merge(test, on=["unique_id", "ds"], how="left")
        fcst = fcst.fillna(-1)
        logging.info(f"Forecast completed for unique_id: {uid}")

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
        logging.info(f"Model summary for best model: {mod_summary}")

        uid_results = {
            **err.to_dict(),
            **mod_summary,
            **err_auto_,
            "best_config": best_model_name,
            "auto_config": arima_config,
            "dataset": f"{data_name},{group}",
            "unique_id": uid,
        }

        # Store results
        results[uid] = uid_results

        # Optimized DataFrame construction
        results_df = pd.DataFrame.from_dict(results, orient="index")
        type_dict = {
            col: float
            for col in results_df.columns
            if col not in ["best_config", "dataset", "auto_config", "unique_id"]
        }
        results_df = results_df.astype(type_dict)
        results_df.to_csv(outfile / f"arima,{data_name},{group}.csv", index=False)
        logging.info(f"Results saved for unique_id: {uid}")

    # Final optimized DataFrame construction
    results_df = pd.DataFrame.from_dict(results, orient="index")
    type_dict = {
        col: float
        for col in results_df.columns
        if col not in ["best_config", "dataset", "auto_config", "unique_id"]
    }
    results_df = results_df.astype(type_dict)
    results_df.to_csv(outfile / f"arima,{data_name},{group}.csv", index=False)
    logging.info("Final results saved")
