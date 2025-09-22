from pprint import pprint

import pandas as pd
import numpy as np
import plotnine as p9

from src.meta.arima._data_reader import MetadataReader

DATASET_PAIRS = [
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
    ('M4', 'Monthly'),
    ('M4', 'Quarterly')
]

RESULTS_DIR = 'assets/results/main'
all_results = []

for data_name, group in DATASET_PAIRS:
    print(data_name, group)

    freq_int = 12 if group == 'Monthly' else 4

    mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=freq_int)

    results_df = pd.read_csv(f'{RESULTS_DIR}/{data_name},{group}.csv')

    # X, y, _, _, cv = mdr.read(fill_na_value=-1)
    # df = X.merge(results_df, on='unique_id')
    # df['ETS_delta'] = (df['AutoETS'] - df['MetaARIMA'] < -0.04).astype(int)
    # df['ARIMA_delta'] = (df['ARIMA(2,1,2)(1,0,0)'] - df['MetaARIMA'] < -0.04).astype(int)
    # df['SN_delta'] = (df['SeasonalNaive'] - df['MetaARIMA'])

    all_results.append(results_df)

results_df = pd.concat(all_results, ignore_index=True)
results_df = results_df.drop(columns='unique_id')

print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))
print(results_df.rank(axis=1, na_option='bottom').mean())

s = results_df.drop(columns=['AutoARIMA2']).median(numeric_only=True).sort_values()

df = s.reset_index()
df.columns = ['Method', 'AverageRank']

p = p9.ggplot(df, p9.aes(x='reorder(Method, AverageRank)', y='AverageRank')) + \
    p9.geom_bar(stat='identity', fill='#4C72B0') + \
    p9.theme_minimal() + \
    p9.theme(axis_text=p9.element_text(size=12)) + \
    p9.labs(x='', y='SMAPE', title='')

p.save('avg_smape.pdf', width=12, height=4.5)
