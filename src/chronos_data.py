from typing import Optional

import datasets
import numpy as np
import pandas as pd

from huggingface_hub import dataset_info


class ChronosDataset:
    # https://github.com/autogluon/fev/blob/main/benchmarks/chronos_zeroshot/results/auto_arima.csv

    DATASET_NAME = 'CHRONOS'
    REPO_ID = 'autogluon/chronos_datasets'

    horizons_map = {
        'm1_quarterly': 2,
        'm1_monthly': 8,
        'm1_yearly': 4,
        'tourism_monthly': 12,
        'tourism_quarterly': 8,
        'tourism_yearly': 4,
    }

    frequency_map = {
        'm1_quarterly': 4,
        'm1_monthly': 12,
        'm1_yearly': 1,
        'tourism_monthly': 12,
        'tourism_quarterly': 4,
        'tourism_yearly': 1,
    }

    context_length = {
        'm1_quarterly': 4,
        'm1_monthly': 12,
        'm1_yearly': 3,
        'tourism_monthly': 24,
        'tourism_quarterly': 8,
        'tourism_yearly': 3,
    }

    frequency_pd = {
        'm1_quarterly': 'Q',
        'm1_monthly': 'M',
        'm1_yearly': 'Y',
        'tourism_monthly': 'M',
        'tourism_quarterly': 'Q',
        'tourism_yearly': 'Y',
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls,
                  group: str,
                  split: str,
                  min_n_instances: Optional[int] = None,
                  id_col: str = 'unique_id',
                  time_col: str = 'ds',
                  target_col: str = 'y'):

        ds = datasets.load_dataset(path=cls.REPO_ID, name=group, split=split)
        ds.set_format("numpy")

        df = cls.to_pandas(ds)
        df = df.rename(columns={'id': id_col,
                                'timestamp': time_col,
                                'target': target_col})
        df = df.drop(columns=['category'])

        if min_n_instances is not None:
            df = cls.prune_uids_by_size(df, min_n_instances)

        return df

    @classmethod
    def load_everything(cls,
                        group: str,
                        split: str,
                        min_n_instances: Optional[int] = None,
                        sample_n_uid: Optional[int] = None):

        if split == 'both':
            # todo ..
            pass

        df = cls.load_data(group=group,
                           split=split,
                           min_n_instances=min_n_instances)

        horizon = cls.horizons_map.get(group)
        n_lags = cls.context_length.get(group)
        freq_str = cls.frequency_pd.get(group)
        freq_int = cls.frequency_map.get(group)

        if sample_n_uid is not None:
            assert isinstance(df, pd.DataFrame)
            df = cls.sample_first_uids(df, sample_n_uid)

        return df, horizon, n_lags, freq_str, freq_int

    @staticmethod
    def prune_uids_by_size(df: pd.DataFrame,
                           min_n_instances: int,
                           id_col: str = 'unique_id'):
        large_ts = df[id_col].value_counts() >= min_n_instances
        large_ts_uid = large_ts[large_ts].index.tolist()

        df = df.query(f'{id_col}== @large_ts_uid').reset_index(drop=True)

        return df

    @staticmethod
    def sample_first_uids(df: pd.DataFrame, n_uid: int, id_col: str = 'unique_id'):
        uid_sample = df[id_col].unique()[:n_uid].tolist()
        df = df.query(f'{id_col}==@uid_sample').reset_index(drop=True)

        return df

    @staticmethod
    def dummify_series(df, id_col: str = 'unique_id', target_col: str = 'y'):
        df_uid = df.copy().groupby(id_col)

        dummied_l = []
        for g, uid_df in df_uid:
            uid_df[target_col] = range(uid_df.shape[0])

            dummied_l.append(uid_df)

        dummy_df = pd.concat(dummied_l, axis=0).reset_index(drop=True)

        return dummy_df

    @staticmethod
    def get_uid_tails(df, tail_size: int, id_col: str = 'unique_id'):
        df_list = []
        for g, df_ in df.groupby(id_col):
            df_list.append(df_.tail(tail_size))

        tail_df = pd.concat(df_list, axis=0).reset_index(drop=True)

        return tail_df

    @staticmethod
    def time_wise_split(df: pd.DataFrame,
                        horizon: int,
                        id_col: str = 'unique_id',
                        time_col: str = 'ds'):
        df_by_unq = df.groupby(id_col)

        train_l, test_l = [], []
        for g, df_ in df_by_unq:
            df_ = df_.sort_values(time_col)

            train_df_g = df_.head(-horizon)
            test_df_g = df_.tail(horizon)

            train_l.append(train_df_g)
            test_l.append(test_df_g)

        train_df = pd.concat(train_l).reset_index(drop=True)
        test_df = pd.concat(test_l).reset_index(drop=True)

        return train_df, test_df

    @staticmethod
    def get_chronos_datasets_names(repo_id='autogluon/chronos_datasets'):
        ds_info = dataset_info(repo_id)

        dataset_names = np.unique([s.rfilename.split('/')[0] for s in ds_info.siblings])
        dataset_names = [x for x in dataset_names if x not in ['.gitattributes', 'README.md']]

        return dataset_names

    @staticmethod
    def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
        """Convert dataset to long data frame format."""
        sequence_columns = [col for col in ds.features if isinstance(ds.features[col], datasets.Sequence)]
        return ds.to_pandas().explode(sequence_columns).infer_objects()
