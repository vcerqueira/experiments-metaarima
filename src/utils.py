import os
import re
from pprint import pprint

import numpy as np
import pandas as pd
import plotnine as p9

RESULTS_DIR = 'assets/results/main'

THEME = p9.theme_538(base_family='Palatino', base_size=12) + \
        p9.theme(plot_margin=.025,
                 panel_background=p9.element_rect(fill='white'),
                 plot_background=p9.element_rect(fill='white'),
                 legend_box_background=p9.element_rect(fill='white'),
                 strip_background=p9.element_rect(fill='white'),
                 legend_background=p9.element_rect(fill='white'),
                 # axis_text_x=p9.element_text(size=9, angle=0),
                 axis_text_y=p9.element_text(size=9),
                 legend_title=p9.element_blank())


def read_results(file_path: str = RESULTS_DIR) -> pd.DataFrame:
    results_list = os.listdir(file_path)

    all_results = []
    for f in results_list:
        print(f)
        df = pd.read_csv(f'{file_path}/{f}')
        ds = f.split('.csv')[0]
        freq = ds.split('_')[-1]

        df['Dataset'] = ds
        df['Frequency'] = freq

        all_results.append(df)

    df = pd.concat(all_results, ignore_index=True)

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
