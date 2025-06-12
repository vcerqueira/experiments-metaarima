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
    all_results.append(results_df)

results_df = pd.concat(all_results, ignore_index=True)
results_df = results_df.drop(columns='unique_id')

print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))
print(results_df.rank(axis=1, na_option='bottom').mean())
print(results_df.dropna().mean())
print(results_df.dropna().median())
print(results_df.dropna().rank(axis=1).mean())



print(results_df.loc[results_df['AutoARIMA2'].isna(),:].mean())
print(results_df.loc[results_df['AutoARIMA2'].isna(),:].median())


