from pprint import pprint

import pandas as pd
import numpy as np

from src.meta.arima._data_reader import MetadataReader

data_pairs = [
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
    ('M4', 'Monthly'),
    ('M4', 'Quarterly')
]

RESULTS_DIR = 'assets/results/main'
all_results = []

for data_name, group in data_pairs:
    print(data_name, group)

    freq_int = 12 if group == 'Monthly' else 4

    mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=freq_int)
    X, y, _, _, cv = mdr.read(fill_na_value=-1)

    results_df = pd.read_csv(f'{RESULTS_DIR}/{data_name},{group}.csv')

    df = X.merge(results_df, on='unique_id')
    # df['ETS_delta'] = 100 * ((df['AutoETS'] - df['MetaARIMA']) / df['MetaARIMA'])
    df['ETS_delta'] = (df['AutoETS'] - df['MetaARIMA'] < -0.02).astype(int)
    df['ARIMA_delta'] = (df['ARIMA(2,1,2)(1,0,0)'] - df['MetaARIMA'] < -0.02).astype(int)
    # df['ARIMA_delta'] = 100 * ((df['ARIMA(2,1,2)(1,0,0)'] - df['MetaARIMA']) / df['MetaARIMA'])

    all_results.append(df)

results_df = pd.concat(all_results, ignore_index=True)
results_df = results_df.drop(columns='unique_id')

print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))
print(results_df.rank(axis=1, na_option='bottom').mean())
print(results_df.dropna().mean())
print(results_df.dropna().median())
print(results_df.dropna().rank(axis=1).mean())

# results_df.corr(method='pearson')['ETS_delta'].sort_values()
# results_df.corr(method='kendall')['ETS_delta'].sort_values()
# results_df.corr(method='kendall')['ARIMA_delta'].sort_values()
# results_df.corr()['ARIMA_delta']

print(results_df.loc[results_df['AutoARIMA2'].isna(), :].mean())
print(results_df.loc[results_df['AutoARIMA2'].isna(), :].median())
methods = ['MetaARIMA','AutoARIMA','ARIMA(2,1,2)(1,0,0)','AutoETS']

df2 = results_df.copy()
# df2 = df2.loc[df2['seasonal_strength'] > 0.75, :]
df2 = df2.loc[df2['series_length'] > 700, :]
# df2 = df2.loc[df2['lumpiness'] > 1, :]
# df2 = df2.loc[df2['entropy'] > .75, :]
# df2 = df2.loc[df2['nonlinearity'] < 1, :]
# df2 = df2.loc[df2['spike'] > 6.953706e-06, :]
# print(df2.mean(numeric_only=True))
print(df2.mean(numeric_only=True)[methods])
#results_df.mean(numeric_only=True)[methods]
# print(df2.median(numeric_only=True))
# results_df['lumpiness'].describe()
# results_df['entropy'].describe()
# results_df['nonlinearity'].describe()
# results_df['spike'].describe()