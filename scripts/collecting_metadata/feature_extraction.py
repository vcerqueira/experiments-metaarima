from pathlib import Path

from tsfeatures import tsfeatures

from src.chronos_data import ChronosDataset

GROUP = 'monash_m1_monthly'

df, horizon, n_lags, _, seas_len = ChronosDataset.load_everything(GROUP)

train, _ = ChronosDataset.time_wise_split(df, horizon=horizon)

if __name__ == '__main__':
    feat_df = tsfeatures(train, freq=seas_len, threads=6)

    outfile = Path(__file__).parent.parent.parent / 'assets' / 'features'
    print(outfile.absolute())

    feat_df.to_csv(f'{outfile}/features,{GROUP}.csv', index=False)
    print(feat_df)
