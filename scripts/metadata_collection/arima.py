import sys
import os
import ujson as json  # for speedup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

os.environ["NIXTLA_ID_AS_COL"] = "1"

from pprint import pprint
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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

print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)

train, test = data_loader.train_test_split(df, horizon=horizon)

ORDER_MAX = {"AR": 4, "I": 1, "MA": 4, "S_AR": 1, "S_I": 1, "S_MA": 1}
models = MetaARIMAUtils.get_models_sf(season_length=freq_int, max_config=ORDER_MAX)
print(len(models))

PREV_RESULTS_CSV = ["arima,M4,Monthly_.csv"]


def process_uid(uid, uid_df, result_files, freq_int, freq_str, horizon, test):
    if uid in result_files:
        return None

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

    arima_config = MetaARIMAUtils.get_model_order(
        sf_auto.fitted_[0][0].model_, as_alias=True, alias_freq=freq_int
    )

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
    best_model = sf.fitted_.flatten()[err.iloc[0]]

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
        r = pd.read_csv(f"{outfile}/{file}")
        result_files += r["unique_id"].values.tolist()

    results = {}
    df_grouped = train.groupby("unique_id")

    max_workers = os.cpu_count()  # available CPU cores
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_uid,
                uid,
                uid_df,
                result_files,
                freq_int,
                freq_str,
                horizon,
                test,
            ): uid
            for uid, uid_df in df_grouped
        }

        for future in as_completed(futures):
            uid_results = future.result()
            if uid_results:
                results[uid_results["unique_id"]] = uid_results

    results_df = pd.DataFrame.from_dict(results).T
    type_dict = {
        col: float
        for col in results_df.columns
        if col not in ["best_config", "dataset", "auto_config", "unique_id"]
    }
    results_df = results_df.astype(type_dict)

    results_df.to_csv(f"{outfile}/arima,{data_name},{group}.csv", index=False)
