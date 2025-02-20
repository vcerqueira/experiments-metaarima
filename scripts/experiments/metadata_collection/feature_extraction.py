from pathlib import Path

from tsfeatures import tsfeatures

from src.load_data.config import DATASETS, DATA_GROUPS, GROUP_IDX

data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=3)

train, _ = data_loader.train_test_split(df, horizon=horizon)

if __name__ == '__main__':
    feat_df = tsfeatures(train, freq=freq_int, threads=6)

    outfile = Path(__file__).parent.parent.parent.parent / 'assets' / 'features'

    feat_df.to_csv(f'{outfile}/train_feature_set_{data_name},{group}.csv', index=False)
    print(feat_df)
