from pprint import pprint

import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset, dataset_names

from src.load_data.base import LoadDataset


# pprint(dataset_names)


class GluontsDataset(LoadDataset):
    DATASET_NAME = 'GLUONTS'

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

    min_samples = {
        'm1_quarterly': 22,
        'm1_monthly': 52,
        'm1_yearly': 10,
        'tourism_monthly': 36,
        'tourism_quarterly': 16,
        'tourism_yearly': 10,
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
                  group,
                  min_n_instances=None,
                  extended=False):

        dataset = get_dataset(group, regenerate=False)
        train_list = dataset.train

        df_list = []
        for i, series in enumerate(train_list):
            s = pd.Series(
                series["target"],
                index=pd.date_range(
                    start=series["start"].to_timestamp(),
                    freq=series["start"].freq,
                    periods=len(series["target"]),
                ),
            )

            if group == 'australian_electricity_demand':
                s = s.resample('W').sum()

            s_df = s.reset_index()
            s_df.columns = ['ds', 'y']
            s_df['unique_id'] = f'ID{i}'

            df_list.append(s_df)

        df = pd.concat(df_list).reset_index(drop=True)
        df = df[['unique_id', 'ds', 'y']]

        if min_n_instances is not None:
            df = cls.prune_df_by_size(df, min_n_instances)

        return df

# df, *_ = GluontsDataset.load_everything('m1_yearly')
# df['unique_id'].nunique()
