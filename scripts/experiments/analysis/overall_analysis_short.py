from pprint import pprint

import pandas as pd
import numpy as np
import plotnine as p9

from src.utils import read_results

DATASET_PAIRS = [
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
    ('M3', 'Monthly'),
    ('M4', 'Monthly'),
    ('M4', 'Quarterly')
]

results_df = read_results()

print(results_df.drop(columns='Dataset').mean(numeric_only=True))
print(results_df.drop(columns='Dataset').median(numeric_only=True))
print(results_df.drop(columns='Dataset').rank(axis=1, na_option='bottom').mean())

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(results_df.groupby('Dataset').mean(numeric_only=True).T)
print(results_df.groupby('Dataset').median(numeric_only=True).T)
# results_df.groupby('Dataset').rank(axis=1, na_option='bottom').mean().T


s = results_df.drop(columns=['AutoARIMA2']).median(numeric_only=True).sort_values()

df = s.reset_index()
df.columns = ['Method', 'AverageRank']

p = p9.ggplot(df, p9.aes(x='reorder(Method, AverageRank)', y='AverageRank')) + \
    p9.geom_bar(stat='identity', fill='#4C72B0') + \
    p9.theme_minimal() + \
    p9.theme(axis_text=p9.element_text(size=12)) + \
    p9.labs(x='', y='SMAPE', title='')

p.save('avg_smape.pdf', width=12, height=4.5)
