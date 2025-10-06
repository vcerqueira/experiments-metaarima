import pandas as pd
import plotnine as p9

from src.meta.arima._data_reader import MetadataReader
from src.utils import THEME

K = 25

mdr = MetadataReader(dataset_name='M4', group='Monthly', freq_int=12)
X, y, _, _, cv = mdr.read(fill_na_value=-1)

cv_cols = cv.columns.tolist()
x_cols = X.columns.tolist()
y_cols = y.columns.tolist()
unique_cv_cols = [col for col in cv_cols if col not in x_cols and col not in y_cols]

cv[unique_cv_cols]
cv[unique_cv_cols].iloc[0]
cv.loc[:, cv.columns.str.startswith('coef')].iloc[0]

# boxplot on rank - top K

config_ranks = y.rank(axis=1).astype(int)
config_avg_rank = config_ranks.mean()
top_k_configs = config_avg_rank.sort_values(ascending=True).index[:K].tolist()
bot_k_configs = config_avg_rank.sort_values(ascending=False).index[:K].tolist()
selected_configs = top_k_configs + bot_k_configs
selected_config_avg_rank = config_avg_rank[selected_configs].sort_values().index.tolist()

config_ranks_mt = config_ranks[selected_configs].melt()
config_ranks_mt['variable'] = config_ranks_mt['variable'].str.replace('ARIMA', '')
config_ranks_mt['variable'] = pd.Categorical(config_ranks_mt['variable'],
                                             categories=[x.replace('ARIMA', '') for x in selected_config_avg_rank])

plot = p9.ggplot(config_ranks_mt, p9.aes(**{'x': 'variable',
                                            'y': 'value'})) + \
       THEME + \
       p9.theme(plot_margin=0.015,
                axis_text_y=p9.element_text(size=12),
                axis_text_x=p9.element_text(size=12, angle=60),
                legend_title=p9.element_blank(),
                legend_position=None) + \
       p9.geom_boxplot(show_legend=False) + \
       p9.xlab('') + \
       p9.ylab('SMAPE Rank')

plot_name = 'assets/results/plots/config_rank_dist.pdf'

plot.save(plot_name, width=12, height=5)

# number of wins p configs
(config_ranks == 1).sum().sort_values(ascending=False)[:25]

## ... not sure this brings any more value than what the previous one already shows


# corr with best
corr_scores = y.corr()
corr_scr = corr_scores['ARIMA(1,1,1)(0,1,1)[12]'].sort_values(ascending=False)

top_50 = corr_scr.head(75)
bottom_50 = corr_scr.tail(75)
filtered_corr = pd.concat([top_50, bottom_50])

df = filtered_corr.reset_index()
df.columns = ['Configuration', 'Correlation']

p = p9.ggplot(df, p9.aes(x='reorder(Configuration, Correlation)', y='Correlation')) + \
    p9.geom_bar(stat='identity', fill='#4C72B0') + \
    THEME + \
    p9.theme(plot_margin=0.015,
             axis_text_y=p9.element_text(size=12),
             axis_text_x=p9.element_blank(),
             legend_title=p9.element_blank(),
             legend_position=None) + \
    p9.labs(x='', y='Pearson Correlation', title='')

plot_name = 'assets/results/plots/config_corr_best.pdf'
p.save(plot_name, width=12, height=4.5)

coef_cols = [col for col in cv.columns if col.startswith('coef_')]
cv_with_rank = cv.copy()
cv_with_rank['avg_rank'] = config_ranks.mean()
cv_with_rank['performance_tier'] = pd.qcut(cv_with_rank['avg_rank'],
                                           q=4,
                                           labels=['Top 25%', 'Q2', 'Q3', 'Bottom 25%'])

# Average coefficient magnitudes by tier
coef_magnitude = cv_with_rank.groupby('performance_tier')[coef_cols].apply(
    lambda x: x.abs().mean()
).T
coef_magnitude['coef_type'] = coef_magnitude.index.str.extract(r'coef_(\w+)\d*')[0]
coef_mag_mt = coef_magnitude.reset_index().melt(
    id_vars=['index', 'coef_type'],
    var_name='tier',
    value_name='magnitude'
)

# Plot
p = p9.ggplot(coef_mag_mt, p9.aes(x='coef_type', y='magnitude', fill='tier')) + \
    p9.geom_bar(stat='identity', position='dodge') + \
    THEME + \
    p9.labs(x='Coefficient Type', y='Average Absolute Magnitude')

p.save('test.pdf', width=12, height=4.5)


cv_with_rank['n_coefs'] = cv[coef_cols].notna().sum(axis=1)
complexity_plot = p9.ggplot(cv_with_rank,
                            p9.aes(x='n_coefs', y='avg_rank')) + \
                  p9.geom_point(alpha=0.3) + \
                  p9.geom_smooth(method='loess') + \
                  THEME + \
                  p9.labs(x='Number of Coefficients', y='Average Rank')
complexity_plot.save('test.pdf', width=12, height=4.5)


##


import numpy as np

# Create configuration-level summary
config_avg_rank = config_ranks.mean(axis=0)  # avg rank per config
config_summary = pd.DataFrame({
    'config': config_avg_rank.index,
    'avg_rank': config_avg_rank.values,
    'median_rank': config_ranks.median(axis=0).values,
    'rank_std': config_ranks.std(axis=0).values,
    'win_rate': (config_ranks == 1).sum(axis=0).values / len(config_ranks)
})

# Parse ARIMA orders from config names
def parse_arima(config_str):
    """Extract (p,d,q)(P,D,Q)[m] from ARIMA config string"""
    import re
    match = re.search(r'ARIMA\((\d+),(\d+),(\d+)\)\((\d+),(\d+),(\d+)\)\[(\d+)\]', config_str)
    if match:
        return tuple(map(int, match.groups()))
    return (None,) * 7

config_summary[['p', 'd', 'q', 'P', 'D', 'Q', 'm']] = \
    config_summary['config'].apply(lambda x: pd.Series(parse_arima(x)))

config_summary['total_params'] = config_summary['p'] + config_summary['q'] + \
                                 config_summary['P'] + config_summary['Q']
config_summary['seasonal'] = (config_summary['P'] + config_summary['Q']) > 0

# Create performance tiers
config_summary['performance_tier'] = pd.qcut(config_summary['avg_rank'],
                                              q=4,
                                              labels=['Top 25%', 'Q2', 'Q3', 'Bottom 25%'],
                                              duplicates='drop')

# 1. Model complexity vs performance
complexity_plot = p9.ggplot(config_summary.dropna(),
                            p9.aes(x='total_params', y='avg_rank')) + \
                  p9.geom_point(alpha=0.5, size=2) + \
                  p9.geom_smooth(method='loess', color='red') + \
                  THEME + \
                  p9.labs(x='Total Parameters (p+q+P+Q)',
                         y='Average Rank',
                         title='Model Complexity vs Performance')

complexity_plot.save('assets/results/plots/complexity_vs_rank.pdf', width=8, height=6)

# 2. Seasonal vs non-seasonal
seasonal_plot = p9.ggplot(config_summary.dropna(),
                          p9.aes(x='seasonal', y='avg_rank')) + \
                p9.geom_boxplot() + \
                THEME + \
                p9.labs(x='Has Seasonal Component',
                       y='Average Rank',
                       title='Seasonal vs Non-Seasonal Models')

seasonal_plot.save('assets/results/plots/seasonal_vs_rank.pdf', width=6, height=6)

# 3. Differencing order analysis
diff_plot = p9.ggplot(config_summary.dropna(),
                      p9.aes(x='factor(d)', y='avg_rank', fill='factor(D)')) + \
            p9.geom_boxplot() + \
            THEME + \
            p9.labs(x='Non-seasonal Differencing (d)',
                   y='Average Rank',
                   fill='Seasonal\nDifferencing (D)')

diff_plot.save('assets/results/plots/differencing_vs_rank.pdf', width=8, height=6)

# 4. AR order vs MA order
ar_ma_plot = p9.ggplot(config_summary.dropna(),
                       p9.aes(x='p', y='q', color='avg_rank', size='win_rate')) + \
             p9.geom_point(alpha=0.7) + \
             p9.scale_color_gradient(low='darkgreen', high='red') + \
             THEME + \
             p9.labs(x='AR Order (p)',
                    y='MA Order (q)',
                    color='Avg Rank',
                    size='Win Rate')

ar_ma_plot.save('assets/results/plots/ar_ma_orders.pdf', width=8, height=6)

# 5. Win rate distribution
win_plot = p9.ggplot(config_summary.nsmallest(30, 'avg_rank'),
                     p9.aes(x='reorder(config, -win_rate)', y='win_rate')) + \
           p9.geom_bar(stat='identity', fill='steelblue') + \
           THEME + \
           p9.theme(axis_text_x=p9.element_text(angle=90, hjust=1, size=8)) + \
           p9.labs(x='', y='Win Rate', title='Top 30 Configurations by Win Rate')

win_plot.save('assets/results/plots/win_rates.pdf', width=10, height=6)

# Print summary statistics
print("\n=== Model Order Statistics by Performance Tier ===")
print(config_summary.groupby('performance_tier')[['p', 'd', 'q', 'P', 'D', 'Q', 'total_params']].mean())

print("\n=== Top 10 Configurations ===")
print(config_summary.nsmallest(10, 'avg_rank')[['config', 'avg_rank', 'median_rank', 'win_rate']])