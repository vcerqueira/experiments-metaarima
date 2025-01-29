import os
from pprint import pprint
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from tsfeatures import tsfeatures

from utils.load_data.config import DATASETS, DATA_GROUPS, GROUP_IDX

data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=5)
df = data_loader.get_uid_tails(df, tail_size=100)

train, _ = data_loader.train_test_split(df, horizon=horizon)

if __name__ == '__main__':

    df_grouped = train.groupby('unique_id')
    for uid, uid_df in df_grouped:
        print(data_name, group, uid)
        # df = ds.query('unique_id=="Y1"')

        feat_df = tsfeatures(uid_df, freq=12, threads=1)
        print(feat_df)
