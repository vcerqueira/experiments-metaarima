from pprint import pprint

import pandas as pd
import plotnine as p9

DATASET_PAIRS = [
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
    ('M4', 'Quarterly')
]

RESULTS_DIR = 'assets/results/sensitivity'

results_df = pd.read_csv(f'{RESULTS_DIR}/transfer,M4,Monthly.csv')

print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))
print(results_df.rank(axis=1, na_option='bottom').mean())
