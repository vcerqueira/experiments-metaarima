"""Shared evaluation loop for MetaARIMA against one foundation model."""

import argparse
import random
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive
from utilsforecast.losses import mase

from src.chronos_data import ChronosDataset
from src.config import RANDOM_SEED
from src.meta.arima._data_reader import ModelIO

REPO_ROOT = Path(__file__).resolve().parents[2]
MONTHLY_TARGETS = (
    "monash_m1_monthly",
    "monash_m3_monthly",
    "monash_tourism_monthly",
)


def dataset_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--source",
        default="m4_monthly",
        help="M4 dataset of the saved metamodel. TSFM comparisons use m4_monthly.",
    )
    parser.add_argument(
        "--target",
        default="monash_tourism_monthly",
        help="Dataset to forecast. Paper comparisons use " + ", ".join(MONTHLY_TARGETS) + ".",
    )
    return parser


def seed_runtime():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(RANDOM_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_evaluation(target, source, model_name, prepare_forecaster, output_subdir):
    """Forecast every series with MetaARIMA, two baselines, and one foundation model.

    prepare_forecaster(horizon, freq) returns a function
    forecast_one(history_df, horizon, freq, future_ds) -> DataFrame
    with columns unique_id, ds, and model_name.
    """
    seed_runtime()
    filename = REPO_ROOT / f"assets/trained_metaarima_{source}_catboost.joblib.gz"
    meta_arima = ModelIO.load_model(filename)

    df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
    train, test = ChronosDataset.time_wise_split(df, horizon)
    forecast_one = prepare_forecaster(horizon, freq)

    sf_models = [AutoARIMA(season_length=seas_len), SeasonalNaive(season_length=seas_len)]
    model_names = ["MetaARIMA", "AutoARIMA", "SeasonalNaive", model_name]

    results, predictions = [], []
    for uid in train["unique_id"].unique().tolist():
        print(uid)
        df_uid_tr = train.query(f'unique_id=="{uid}"').reset_index(drop=True)
        df_uid_ts = test.query(f'unique_id=="{uid}"').reset_index(drop=True)
        if df_uid_ts.isna().any()["y"]:
            continue

        meta_arima.fit(df_uid_tr, freq=freq, seas_length=seas_len)
        fcst_ma = meta_arima.predict(h=horizon)

        sf = StatsForecast(models=deepcopy(sf_models), freq=freq)
        sf.fit(df_uid_tr)
        fcst_aa = sf.forecast(h=horizon)

        fcst_tsfm = forecast_one(df_uid_tr, horizon, freq, df_uid_ts["ds"])
        fcst_tsfm = fcst_tsfm[["unique_id", "ds", model_name]]

        uid_test = df_uid_ts.merge(fcst_ma, on=["unique_id", "ds"])
        uid_test = uid_test.merge(fcst_aa, on=["unique_id", "ds"])
        uid_test = uid_test.merge(fcst_tsfm, on=["unique_id", "ds"])

        err = mase(
            df=uid_test,
            models=model_names,
            seasonality=seas_len,
            train_df=df_uid_tr,
        )
        pprint(err)

        predictions.append(uid_test)
        results.append(err)
        results_df = pd.concat(results, ignore_index=True)
        print(results_df.mean(numeric_only=True))
        print(results_df.median(numeric_only=True))

    results_df = pd.concat(results, ignore_index=True)
    predictions_df = pd.concat(predictions, ignore_index=True)
    print(results_df.mean(numeric_only=True))
    print(results_df.median(numeric_only=True))

    output_dir = REPO_ROOT / "assets" / "results" / "tsfm" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / f"scores,{target}.csv", index=False)
    predictions_df.to_csv(output_dir / f"predictions,{target}.csv", index=False)
    print(output_dir)
