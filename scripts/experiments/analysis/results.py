from pprint import pprint

import pandas as pd
import numpy as np


from src.meta.arima._data_reader import MetadataReader

data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
# data_name, group = 'M4', 'Quarterly'
print(data_name, group)

RESULTS_DIR = 'assets/results/main'
freq_int = 12

mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=freq_int)

X, y, _, _, cv = mdr.read(fill_na_value=-1)

results_df = pd.read_csv(f'{RESULTS_DIR}/{data_name},{group}.csv')

# res = results_df.merge(cv[['auto_config']], on='unique_id')
#
# res['d'] = res['MetaARIMA']-res['AutoARIMA']
# res = res.sort_values('d', ascending=False)
#
# res.loc[res['d'] > 0.025,:]['auto_config'].value_counts()

print(results_df.mean(numeric_only=True))
print(results_df.dropna().mean(numeric_only=True))





