"""Forecast with TimesFM and score it against MetaARIMA."""

import numpy as np
import pandas as pd
import torch
from transformers import TimesFmModelForPrediction

from common import dataset_parser, run_evaluation

MODEL_ID = "google/timesfm-2.0-500m-pytorch"
MODEL_REVISION = "dc2443792ce5516872b89b37cf1bc058c3bf0c10"
MODEL_NAME = "TimesFM"
CONTEXT_LENGTH = None


def timesfm_frequency_id(freq):
    """Map a pandas or StatsForecast frequency string to a TimesFM frequency id."""
    freq = str(freq).upper()
    if freq.endswith("MS"):
        return 1
    if freq.endswith(("H", "T", "MIN", "D", "B", "U", "S")):
        return 0
    if freq.endswith(("W", "M")) or freq.startswith("W-") or (freq.startswith("M") and len(freq) == 2):
        return 1
    if freq.endswith(("Y", "Q", "A")) or freq.startswith(("Y-", "Q-", "A-")):
        return 2
    raise ValueError(f"Invalid TimesFM frequency: {freq}")


def timesfm_predict_df(history_df, model, prediction_length, freq, future_ds=None):
    """Forecast a long dataframe with columns unique_id, ds, y using TimesFM."""
    required_cols = ["unique_id", "ds", "y"]
    missing_cols = [col for col in required_cols if col not in history_df.columns]
    if missing_cols:
        raise ValueError(f"history_df is missing columns: {missing_cols}")

    timesfm_input = history_df[required_cols].copy()
    timesfm_input["ds"] = pd.to_datetime(timesfm_input["ds"])
    timesfm_input = timesfm_input.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    ids = []
    grouped_inputs = []
    forecast_input = []
    for uid, uid_df in timesfm_input.groupby("unique_id", sort=False):
        ids.append(uid)
        grouped_inputs.append(uid_df)
        forecast_input.append(
            torch.tensor(uid_df["y"].to_numpy(dtype=np.float32), dtype=torch.float32).to(model.device)
        )

    frequency_input = torch.tensor(
        [timesfm_frequency_id(freq)] * len(forecast_input),
        dtype=torch.long,
    ).to(model.device)

    with torch.no_grad():
        if CONTEXT_LENGTH is None:
            outputs = model(forecast_input, frequency_input)
        else:
            outputs = model(
                forecast_input,
                frequency_input,
                forecast_context_len=CONTEXT_LENGTH,
            )
        point_forecast = outputs.mean_predictions.float().cpu().numpy()

    frames = []
    future_ds = None if future_ds is None else pd.to_datetime(pd.Series(future_ds))
    for i, (uid, uid_history) in enumerate(zip(ids, grouped_inputs)):
        y_pred = np.asarray(point_forecast[i, :prediction_length]).reshape(-1)
        if future_ds is not None and len(ids) == 1:
            ds_values = future_ds.iloc[: len(y_pred)].to_numpy()
        else:
            last_ds = uid_history["ds"].max()
            ds_values = pd.date_range(
                start=last_ds,
                periods=prediction_length + 1,
                freq=freq,
            )[1: len(y_pred) + 1]

        frames.append(pd.DataFrame({
            "unique_id": uid,
            "ds": ds_values,
            MODEL_NAME: y_pred[: len(ds_values)],
        }))

    return pd.concat(frames, ignore_index=True)


def prepare_forecaster(horizon, freq):
    # CPU inference. device_map="auto" will use a GPU when one is available.
    model = TimesFmModelForPrediction.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="sdpa",
        device_map="cpu",
    )

    def forecast_one(history_df, horizon, freq, future_ds):
        return timesfm_predict_df(
            history_df,
            model=model,
            prediction_length=horizon,
            freq=freq,
            future_ds=future_ds,
        )

    return forecast_one


if __name__ == "__main__":
    args = dataset_parser("Evaluate TimesFM against a saved MetaARIMA model.").parse_args()
    run_evaluation(
        target=args.target,
        source=args.source,
        model_name=MODEL_NAME,
        prepare_forecaster=prepare_forecaster,
        output_subdir="timesfm",
    )
