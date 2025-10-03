"""

3. barplot ou tabela dos resultados das variantes


"""
from pprint import pprint

import pandas as pd
import plotnine as p9

DATASET_PAIRS = [
    ('M3', 'Monthly'),
    #('Tourism', 'Monthly'),
]

RESULTS_DIR = 'assets/results/sensitivity'
all_results = []

for data_name, group in DATASET_PAIRS:
    print(data_name, group)
    freq_int = 12

    results_df = pd.read_csv(f'{RESULTS_DIR}/ablation,{data_name},{group}.csv')

    all_results.append(results_df)

results_df = pd.concat(all_results, ignore_index=True)

print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))
print(results_df.rank(axis=1, na_option='bottom').mean())

s = results_df.median(numeric_only=True).sort_values()

df = s.reset_index()
df.columns = ['Method', 'AverageRank']

p = p9.ggplot(df, p9.aes(x='reorder(Method, AverageRank)', y='AverageRank')) + \
    p9.geom_bar(stat='identity', fill='#4C72B0') + \
    p9.theme_minimal() + \
    p9.theme(axis_text=p9.element_text(size=12)) + \
    p9.labs(x='', y='SMAPE', title='')

p.save('avg_smape.pdf', width=12, height=4.5)
