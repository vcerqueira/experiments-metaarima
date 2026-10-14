"""Summarise committed foundation-model scores."""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "results" / "tsfm"
ID_COLUMNS = {
    "unique_id",
    "id",
    "series_id",
    "timestamp",
    "time",
    "date",
    "ds",
    "y",
    "target",
    "split",
    "fold",
    "horizon",
}


def load_summary(results_dir):
    rows = []
    for file in sorted(results_dir.glob("*/scores*.csv")):
        df = pd.read_csv(file)
        method_columns = [
            col
            for col in df.columns
            if col not in ID_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not method_columns:
            continue

        dataset = (
            file.stem.replace("scores,", "")
            .replace("scores_", "")
            .replace("scores-", "")
        )
        row = {"file": file.name, "dataset": dataset, "n_series": len(df)}
        row.update(df[method_columns].mean(numeric_only=True).to_dict())
        rows.append(row)

    raw_summary = pd.DataFrame(rows)
    method_columns = [
        col for col in raw_summary.columns if col not in {"file", "dataset", "n_series"}
    ]
    summary = raw_summary.groupby(["file", "dataset"], as_index=False).agg(
        {"n_series": "max", **{col: "mean" for col in method_columns}}
    )
    summary = (
        summary[["dataset", *method_columns]]
        .sort_values("dataset")
        .reset_index(drop=True)
        .drop(columns=["SeasonalNaive"])
        .set_index("dataset")
    )
    return raw_summary, summary


def print_m3_extremes(results_dir):
    dataset = "monash_m3_monthly"
    scores_path = next(results_dir.glob(f"chronos2/scores*{dataset}*.csv"))
    scores = pd.read_csv(scores_path)

    best_ids = pd.Series(
        {
            "Chronos2": scores.loc[scores["Chronos2"].idxmin(), "unique_id"],
            "MetaARIMA": scores.loc[scores["MetaARIMA"].idxmin(), "unique_id"],
        },
        name=dataset,
    )
    print(best_ids.to_string())

    scores["diff"] = scores["Chronos2"] - scores["MetaARIMA"]
    largest_diff = scores.loc[scores["diff"].idxmax()]
    smallest_diff = scores.loc[scores["diff"].idxmin()]
    extremes = pd.DataFrame(
        {
            "unique_id": [largest_diff["unique_id"], smallest_diff["unique_id"]],
            "Chronos2": [largest_diff["Chronos2"], smallest_diff["Chronos2"]],
            "MetaARIMA": [largest_diff["MetaARIMA"], smallest_diff["MetaARIMA"]],
            "Chronos2 - MetaARIMA": [largest_diff["diff"], smallest_diff["diff"]],
        },
        index=["MetaARIMA better", "Chronos2 better"],
    )
    print(extremes.to_string())


if __name__ == "__main__":
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    raw_summary, summary = load_summary(RESULTS_DIR)
    print(summary.round(3).astype(str).to_string())
    print(summary.round(3).astype(str).to_latex(
        caption="Mean MASE on the monthly evaluation datasets.",
        label="tab:tsfm",
    ))
    print(raw_summary.query('dataset=="monash_m3_monthly"').to_string(index=False))
    print_m3_extremes(RESULTS_DIR)
