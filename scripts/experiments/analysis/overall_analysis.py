import warnings

import numpy as np
import pandas as pd
import plotnine as p9

from src.utils import to_latex_tab, read_results

warnings.simplefilter(action='ignore', category=FutureWarning)

TRUNCATE_DIST = True

df = read_results()
df_mt = df.drop(columns=['Dataset', 'Frequency', 'unique_id']).melt()

avg_by_ds = df.groupby(['Dataset']).mean(numeric_only=True).round(4)
# avg_by_ds.loc['Avg.', :] = avg_by_ds.mean().values
avg_by_ds.loc['Avg. Rank', :] = avg_by_ds.rank(axis=1).mean().round(2).values

med_by_ds = df.groupby(['Dataset']).median(numeric_only=True).round(4)
# med_by_ds.loc['Avg.', :] = med_by_ds.mean().values
med_by_ds.loc['Avg. Rank', :] = med_by_ds.rank(axis=1).mean().round(2).values

avg_by_ds_tab = to_latex_tab(avg_by_ds, 4, rotate_cols=True)
print(avg_by_ds_tab)

med_by_ds_tab = to_latex_tab(med_by_ds, 4, rotate_cols=True)
# print(med_by_ds_tab)
