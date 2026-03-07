import re

import pandas as pd
import plotnine as p9

from src.meta.arima._data_reader import MetadataReader
from src.utils import THEME

K = 25

mdr = MetadataReader(group='m4_monthly', freq_int=12)
X, y, _, _, cv = mdr.read(fill_na_value=-1)

cv_cols = cv.columns.tolist()
x_cols = X.columns.tolist()
y_cols = y.columns.tolist()
unique_cv_cols = [col for col in cv_cols if col not in x_cols and col not in y_cols]

# cv[unique_cv_cols]
# cv[unique_cv_cols].iloc[0]
# cv.loc[:, cv.columns.str.startswith('coef')].iloc[0]

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
       p9.ylab('MASE Rank')

plot_name = 'assets/results/plots/config_rank_dist.pdf'

plot.save(plot_name, width=12, height=5)

# number of wins p configs
# (config_ranks == 1).sum().sort_values(ascending=False)[:25]

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

# complexity v avg rank

config_avg_rank = config_ranks.mean(axis=0)  # avg rank per config
config_summary = pd.DataFrame({
    'config': config_avg_rank.index,
    'avg_rank': config_avg_rank.values,
    'median_rank': config_ranks.median(axis=0).values,
    'rank_std': config_ranks.std(axis=0).values,
    'win_rate': (config_ranks == 1).sum(axis=0).values / len(config_ranks)
})


def parse_arima(config_str):
    """Extract (p,d,q)(P,D,Q)[m] from ARIMA config string"""

    match = re.search(r'ARIMA\((\d+),(\d+),(\d+)\)\((\d+),(\d+),(\d+)\)\[(\d+)\]', config_str)
    if match:
        return tuple(map(int, match.groups()))
    return (None,) * 7


config_summary[['p', 'd', 'q', 'P', 'D', 'Q', 'm']] = \
    config_summary['config'].apply(lambda x: pd.Series(parse_arima(x)))

config_summary['total_params'] = config_summary['p'] + config_summary['q'] + \
                                 config_summary['P'] + config_summary['Q']
config_summary['seasonal'] = (config_summary['P'] + config_summary['Q']) > 0

complexity_plot = p9.ggplot(config_summary.dropna(),
                            p9.aes(x='total_params', y='avg_rank')) + \
                  p9.geom_point(alpha=0.5, size=3) + \
                  p9.geom_smooth(method='loess', color='red') + \
                  THEME + \
                  p9.labs(x='Total Parameters (p+q+P+Q)', y='Average Rank', title='')

complexity_plot.save('assets/results/plots/complexity_vs_rank.pdf', width=8, height=6)
