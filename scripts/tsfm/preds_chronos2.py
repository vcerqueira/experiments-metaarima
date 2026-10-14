"""Forecast with Chronos-2 and score it against MetaARIMA."""

from chronos import BaseChronosPipeline

from common import dataset_parser, run_evaluation

MODEL_NAME = "Chronos2"
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def prepare_forecaster(horizon, freq):
    # CPU inference. device_map="cuda" is faster when a GPU is available.
    chronos2 = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        revision=CHRONOS_REVISION,
    )

    def forecast_one(history_df, horizon, freq, future_ds):
        forecast = chronos2.predict_df(
            history_df,
            prediction_length=horizon,
            id_column="unique_id",
            timestamp_column="ds",
            target="y",
        )
        return forecast.rename(columns={"predictions": MODEL_NAME})

    return forecast_one


if __name__ == "__main__":
    args = dataset_parser("Evaluate Chronos-2 against a saved MetaARIMA model.").parse_args()
    run_evaluation(
        target=args.target,
        source=args.source,
        model_name=MODEL_NAME,
        prepare_forecaster=prepare_forecaster,
        output_subdir="chronos2",
    )
