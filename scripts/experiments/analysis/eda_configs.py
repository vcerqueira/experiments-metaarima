import pandas as pd
import plotnine as p9

from src.meta.arima._data_reader import MetadataReader

RESULTS_DIR = 'assets/results/sensitivity'
all_results = []

mdr = MetadataReader(dataset_name='M3', group='Monthly', freq_int=12)
X, y, _, _, cv = mdr.read(fill_na_value=-1)

cv.loc[:,cv.columns.str.startswith('coef')].iloc[0]

corr_scores = y.corr()
corr_scores['ARIMA(0,0,0)(0,0,0)[12]'].sort_values()


s = corr_scores.median(numeric_only=True).sort_values()

df = s.reset_index()
df.columns = ['Method', 'AverageRank']

p = p9.ggplot(df, p9.aes(x='reorder(Method, AverageRank)', y='AverageRank')) + \
    p9.geom_bar(stat='identity', fill='#4C72B0') + \
    p9.theme_minimal() + \
    p9.theme(axis_text=p9.element_text(size=12)) + \
    p9.labs(x='', y='SMAPE', title='')

p.save('avg_smape.pdf', width=12, height=4.5)

# TODO