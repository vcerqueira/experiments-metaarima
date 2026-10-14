"""Forecast with Moirai-2 and score it against MetaARIMA."""

import numpy as np
import pandas as pd
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from common import dataset_parser, run_evaluation

MODEL_ID = "Salesforce/moirai-2.0-R-small"
MODEL_REVISION = "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"
MODEL_NAME = "Moirai2"
CONTEXT_LENGTH = 1680
BATCH_SIZE = 32


def moirai2_predict_df(history_df, predictor, prediction_length, freq, future_ds=None):
    """Forecast a long dataframe with columns unique_id, ds, y using Moirai-2."""
    required_cols = ["unique_id", "ds", "y"]
    missing_cols = [col for col in required_cols if col not in history_df.columns]
    if missing_cols:
        raise ValueError(f"history_df is missing columns: {missing_cols}")

    moirai_input = history_df[required_cols].copy()
    moirai_input["ds"] = pd.to_datetime(moirai_input["ds"])
    moirai_input = moirai_input.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    ids = []
    grouped_inputs = []
    past_targets = []
    for uid, uid_history in moirai_input.groupby("unique_id", sort=False):
        ids.append(uid)
        grouped_inputs.append(uid_history)
        past_targets.append(uid_history["y"].to_numpy(dtype=np.float32))

    moirai_model = getattr(predictor, "prediction_net", predictor)
    quantile_forecasts = moirai_model.predict(past_targets)
    quantile_levels = np.asarray([
        float(str(q).lstrip("p")) / (100 if str(q).startswith("p") else 1)
        for q in moirai_model.module.quantile_levels
    ])
    median_idx = int(np.argmin(np.abs(quantile_levels - 0.5)))
    point_forecasts = quantile_forecasts[:, median_idx, :prediction_length]

    frames = []
    future_ds = None if future_ds is None else pd.to_datetime(pd.Series(future_ds))
    for i, (uid, uid_history) in enumerate(zip(ids, grouped_inputs)):
        y_pred = np.atleast_1d(np.asarray(point_forecasts[i]).squeeze())
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 0]

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
    # CPU inference. Call .cuda() before creating the predictor when a GPU is available.
    module = Moirai2Module.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to("cpu")
    model = Moirai2Forecast(
        module=module,
        prediction_length=horizon,
        context_length=CONTEXT_LENGTH,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=BATCH_SIZE)

    def forecast_one(history_df, horizon, freq, future_ds):
        return moirai2_predict_df(
            history_df,
            predictor=predictor,
            prediction_length=horizon,
            freq=freq,
            future_ds=future_ds,
        )

    return forecast_one


if __name__ == "__main__":
    args = dataset_parser("Evaluate Moirai-2 against a saved MetaARIMA model.").parse_args()
    run_evaluation(
        target=args.target,
        source=args.source,
        model_name=MODEL_NAME,
        prepare_forecaster=prepare_forecaster,
        output_subdir="moirai2",
    )
