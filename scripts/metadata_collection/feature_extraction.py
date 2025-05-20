from pathlib import Path

from tsfeatures import tsfeatures

from src.load_data.config import DATASETS

# data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
data_name, group = 'M4', 'Quarterly'
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=3)

train, _ = data_loader.train_test_split(df, horizon=horizon)

if __name__ == '__main__':
    feat_df = tsfeatures(train, freq=freq_int, threads=6)

    outfile = Path(__file__).parent.parent.parent / 'assets' / 'features'
    print(outfile.absolute())

    feat_df.to_csv(f'{outfile}/features,{data_name},{group}.csv', index=False)
    print(feat_df)
