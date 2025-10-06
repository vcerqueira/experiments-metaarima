import pandas as pd
import plotnine as p9

from src.utils import THEME

PLOT_NAME = 'assets/results/plots/ablation_scores.pdf'

results_df = pd.read_csv('assets/results/sensitivity/lambda,M3,Monthly.csv')

print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))
print(results_df.rank(axis=1, na_option='bottom').mean())

avg_scores = results_df.median(numeric_only=True)

# Extract lambda values and create dataframe for MetaARIMA results
meta_arima_mask = avg_scores.index.str.contains('MetaARIMA')
df_meta = pd.DataFrame({
    'lambda': avg_scores.index[meta_arima_mask].str.extract(r'\((.*?)\)')[0].astype(float),
    'SMAPE': avg_scores[meta_arima_mask].values
})

# Get AutoARIMA score for horizontal line
auto_arima_score = avg_scores[avg_scores.index == 'AutoARIMA'].iloc[0]

# Create line plot dataframe
df = df_meta.sort_values('lambda')

# Add AutoARIMA reference line dataframe
df_ref = pd.DataFrame({'SMAPE': [auto_arima_score]})

p = p9.ggplot(df, p9.aes(x='reorder(Method, SMAPE)', y='SMAPE')) + \
    THEME + \
    p9.geom_bar(stat='identity', fill='#4C72B0') + \
    p9.theme_minimal() + \
    p9.theme(axis_text=p9.element_text(size=12)) + \
    p9.labs(x='', y='SMAPE', title='')

p.save(PLOT_NAME, width=12, height=4.5)
