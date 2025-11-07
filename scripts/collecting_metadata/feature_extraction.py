from pathlib import Path

from tsfeatures import tsfeatures

from src.chronos_data import ChronosDataset

GROUP = 'm4_yearly'

df, horizon, n_lags, _, seas_len = ChronosDataset.load_everything(GROUP)

train, _ = ChronosDataset.time_wise_split(df, horizon=horizon)
dev, _ = ChronosDataset.time_wise_split(train, horizon=horizon)

if __name__ == '__main__':
    feat_df = tsfeatures(dev, freq=seas_len, threads=6)

    outfile = Path(__file__).parent.parent.parent / 'assets' / 'metadata'
    print(outfile.absolute())

    feat_df.to_csv(f'{outfile}/features,{GROUP}.csv', index=False)
    print(feat_df)
