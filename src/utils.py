import os
import re
from pprint import pprint

import numpy as np
import plotnine as p9

from src.meta.arima._data_reader import MetadataReader

import pandas as pd
import plotnine as p9

RESULTS_DIR = 'assets/results/main'

DS_MAPPER = {
    'M4-Monthly': 'M4-M',
    'M4-Quarterly': 'M4-Q',
    'M3-Monthly': 'M3-M',
    'M3-Quarterly': 'M3-Q',
    'Tourism-Monthly': 'T-M',
    'Tourism-Quarterly': 'T-Q',
}

DATASET_PAIRS = [
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
    ('M3', 'Monthly'),
    ('M4', 'Monthly'),
    ('M4', 'Quarterly')
]

THEME = p9.theme_538(base_family='Palatino', base_size=12) + \
        p9.theme(plot_margin=.025,
                 panel_background=p9.element_rect(fill='white'),
                 plot_background=p9.element_rect(fill='white'),
                 legend_box_background=p9.element_rect(fill='white'),
                 strip_background=p9.element_rect(fill='white'),
                 legend_background=p9.element_rect(fill='white'),
                 axis_text_x=p9.element_text(size=9, angle=0),
                 axis_text_y=p9.element_text(size=9),
                 legend_title=p9.element_blank())


def read_results(file_path: str = RESULTS_DIR) -> pd.DataFrame:
    all_results = []
    for data_name, group in DATASET_PAIRS:
        print(data_name, group)

        freq_int = 12 if group == 'Monthly' else 4

        results_df_ = pd.read_csv(f'{file_path}/{data_name},{group}.csv')
        results_df_['Dataset'] = f'{data_name}-{group[0]}'

        # mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=freq_int)
        # X, y, _, _, cv = mdr.read(fill_na_value=-1)
        # print(X)
        # df = X.merge(results_df, on='unique_id')
        # df['ETS_delta'] = (df['AutoETS'] - df['MetaARIMA'] < -0.04).astype(int)
        # df['ARIMA_delta'] = (df['ARIMA(2,1,2)(1,0,0)'] - df['MetaARIMA'] < -0.04).astype(int)
        # df['SN_delta'] = (df['SeasonalNaive'] - df['MetaARIMA'])

        all_results.append(results_df_)

    df = pd.concat(all_results, ignore_index=True)
    df = df.drop(columns='unique_id')

    return df


def to_latex_tab(df, round_to_n, rotate_cols: bool):
    if rotate_cols:
        df.columns = [f'\\rotatebox{{60}}{{{x}}}' for x in df.columns]

    annotated_res = []
    for i, r in df.round(round_to_n).iterrows():
        top_2 = r.sort_values().unique()[:2]
        if len(top_2) < 2:
            raise ValueError('only one score')

        best1 = r[r == top_2[0]].values[0]
        best2 = r[r == top_2[1]].values[0]

        r[r == top_2[0]] = f'\\textbf{{{best1}}}'
        r[r == top_2[1]] = f'\\underline{{{best2}}}'

        annotated_res.append(r)

    annotated_res = pd.DataFrame(annotated_res).astype(str)

    text_tab = annotated_res.to_latex(caption='CAPTION', label='tab:scores_by_ds')

    return text_tab
