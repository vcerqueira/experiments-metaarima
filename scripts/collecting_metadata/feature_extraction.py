import argparse
from pathlib import Path

from tsfeatures import tsfeatures

from src.chronos_data import ChronosDataset

parser = argparse.ArgumentParser(description="Extract tsfeatures for one Chronos dataset.")
parser.add_argument(
    "--source",
    default="monash_car_parts",
    help="Hugging Face config name. Paper metadata uses m4_monthly, m4_quarterly, or m4_yearly.",
)
GROUP = parser.parse_args().source

df, horizon, n_lags, _, seas_len = ChronosDataset.load_everything(GROUP)

train, _ = ChronosDataset.time_wise_split(df, horizon=horizon)
dev, _ = ChronosDataset.time_wise_split(train, horizon=horizon)

if __name__ == '__main__':
    feat_df = tsfeatures(dev.copy(), freq=seas_len, threads=6)

    outfile = Path(__file__).parent.parent.parent / 'assets' / 'metadata'
    outfile.mkdir(parents=True, exist_ok=True)
    print(outfile.absolute())

    if 'unique_id' in feat_df.columns:
        feat_df = feat_df.sort_values('unique_id')
    else:
        feat_df = feat_df.sort_index()
    feat_df.to_csv(f'{outfile}/features-dev,{GROUP}.csv', index=False)
    print(feat_df)
