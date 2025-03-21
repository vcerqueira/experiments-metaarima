import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pathlib import Path
import logging
from tsfeatures import tsfeatures
from src.load_data.config import DATASETS


# data_name, group = 'M3', 'Monthly'
data_name, group = "M3", "Quarterly"
# data_name, group = "Tourism", "Monthly"
# data_name, group = "Tourism", "Quarterly"
# data_name, group = "M4", "Monthly"
# data_name, group = "M4", "Quarterly"
# data_name, group = "M4", "Weekly"
# data_name, group = "M4", "Daily"

data_loader = DATASETS[data_name]

logging.basicConfig(level=logging.INFO)
logging.info(f"Starting processing for dataset: {data_name}, group: {group}")

try:
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, extended=True
    )
    logging.info(f"Data loaded with horizon: {horizon}, frequency: {freq_str}")

    train, _ = data_loader.train_test_split(df, horizon=horizon)

    if __name__ == "__main__":
        outfile = Path(__file__).resolve().parents[2] / "assets" / "features"
        logging.info(f"Saving to {outfile}")

        outfile.mkdir(parents=True, exist_ok=True)
        feat_df = tsfeatures(train, freq=freq_int)

        output_path = outfile / f"train_feature_set_{data_name},{group}.csv"
        feat_df.to_csv(output_path, index=False)
        logging.info(f"Features saved to {output_path}")
        print(feat_df)
except Exception as e:
    logging.error(f"An error occurred: {e}")
