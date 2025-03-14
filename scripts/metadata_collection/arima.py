import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
from scipy.optimize import OptimizeWarning

os.environ["NIXTLA_ID_AS_COL"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pprint import pprint
from pathlib import Path

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive, AutoTheta, AutoETS
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate

from src.load_data.config import DATASETS
from src.arima.meta import MetaARIMAUtils

# data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = "Tourism", "Monthly"
# data_name, group = 'Tourism', 'Quarterly'
data_name, group = "M4", "Monthly"
# data_name, group = 'M4', 'Quarterly'


print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
train, test = data_loader.train_test_split(df, horizon=horizon)

ORDER_MAX = {"AR": 4, "I": 1, "MA": 4, "S_AR": 1, "S_I": 1, "S_MA": 1}
models = MetaARIMAUtils.get_models_sf(season_length=freq_int, max_config=ORDER_MAX)
print(len(models))

PREV_RESULTS_CSV = ["arima,M4,Monthly_.csv"]


def process_series(uid_df_pair):
    """
    Function to process a single series (unique_id) in parallel.
    """
    uid, uid_df = uid_df_pair

    print(data_name, group, uid)
    if uid in result_files:
        return None

    try:
        sf_auto = StatsForecast(
            models=[
                # AutoARIMA(),
                SeasonalNaive(season_length=freq_int),
                AutoETS(season_length=freq_int),
                AutoTheta(),
            ],
            freq=freq_str,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=InterpolationWarning)
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            sf_auto.fit(df=uid_df)

        fcst_auto = sf_auto.predict(h=horizon)

        arima_config = MetaARIMAUtils.get_model_order(
            sf_auto.fitted_[0][0].model_, as_alias=True, alias_freq=freq_int
        )

    except InterpolationWarning:
        print(f"Skipping series {uid} due to InterpolationWarning.")
        return None
    except Exception as e:
        print(f"Error processing {uid}: {e}")
        return None
    sf = StatsForecast(models=models, freq=freq_str)
    try:
        sf.fit(df=uid_df)
    except ValueError:
        return None

    fcst = sf.predict(h=horizon)

    fcst = fcst.merge(test, on=["unique_id", "ds"], how="left")
    fcst_auto = fcst_auto.merge(test, on=["unique_id", "ds"], how="left")
    fcst = fcst.fillna(-1)

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

    uid_results = {
        **err.to_dict(),
        **mod_summary,
        **err_auto_,
        "best_config": best_model_name,
        "auto_config": arima_config,
        "dataset": f"{data_name},{group}",
        "unique_id": uid,
    }

    return uid_results


if __name__ == "__main__":
    outfile = Path(__file__).parent.parent.parent / "assets" / "metadata_cv"

    result_files = []
    for file in PREV_RESULTS_CSV:
        file_path = outfile / file
        if file_path.exists():
            r = pd.read_csv(file_path)
            result_files += r["unique_id"].values.tolist()

    results = {}

    df_grouped = train.groupby("unique_id")

    with ThreadPoolExecutor() as executor:
        future_to_uid_results = {
            executor.submit(process_series, (uid, uid_df)): uid
            for uid, uid_df in df_grouped
        }

        for future in as_completed(future_to_uid_results):
            uid = future_to_uid_results[future]
            try:
                result = future.result()
                if result is not None:
                    results[uid] = result
            except Exception as e:
                print(f"Error processing {uid}: {e}")


results_df_final = pd.DataFrame.from_dict(results).T

type_dict_final = {
    col: float
    for col in results_df_final.columns
    if col not in ["best_config", "dataset", "auto_config", "unique_id"]
}
results_df_final = results_df_final.astype(type_dict_final)

if not results_df_final.empty:
    output_file_final = outfile / f"arima,{data_name},{group}.csv"

    try:
        output_file_final.parent.mkdir(parents=True, exist_ok=True)
        results_df_final.to_csv(output_file_final, index=False)
        print(f"Results saved successfully to {output_file_final}")

    except Exception as e:
        print(f"Error saving final results: {e}")
else:
    print("Warning: No data to save!")
