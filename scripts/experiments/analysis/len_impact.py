import warnings

import numpy as np
import pandas as pd
import plotnine as p9

from src.utils import THEME, read_results

warnings.simplefilter(action='ignore', category=FutureWarning)

df = read_results(read_len=True)
df = df[['MetaARIMA','AutoETS','series_length']]

lens = [50, 75, 100,125, 150, 175, 200]
sc = []
for l_ in lens:
    df_ = df.query(f'series_length  < {l_}')
    x=df_.mean()
    x['support'] = df_.shape[0]/df.shape[0]
    sc.append(x)
print(pd.DataFrame(sc))

df.query('series_length  > 100').mean()
df.query('series_length  < 100').mean()

# df = df.query('series_length  > 50')

# Calculate difference between MetaARIMA and AutoETS
df['diff'] = df['MetaARIMA'] - df['AutoETS']
df['diff'] = 100* ((df['MetaARIMA'] - df['AutoETS']) /df['AutoETS'])
df['diff'] = np.sign(df['diff']) * np.log(np.abs(df['diff']) + 1)
# df['diff'] = np.sign(df['MetaARIMA'] - df['AutoETS']) * np.log(np.abs(df['MetaARIMA'] - df['AutoETS']) + 1)

# Create scatter plot
# plot = (p9.ggplot(df, p9.aes(x='np.log(series_length+1)', y='diff'))
#         + p9.geom_point(alpha=0.5)
#         + p9.geom_smooth(method='loess', color='blue')
#         + p9.geom_hline(yintercept=0, linetype='dashed', color='red')
#         + p9.labs(x='Series Length',
#                  y='MetaARIMA - AutoETS Difference',
#                  title='Performance Difference vs Series Length')
#         + THEME)
#
# plot.save('test.pdf', width=6, height=5)

# Create length bins (10 bins, unnamed)
df['length_bin'] = pd.qcut(df['series_length'], q=5)

# Average performance by bin
perf_by_length = df.groupby('length_bin')[['MetaARIMA', 'AutoETS']].mean()

# Melt for plotting
perf_melted = perf_by_length.reset_index().melt(id_vars='length_bin', 
                                               var_name='Method', 
                                               value_name='SMAPE')

plot = (p9.ggplot(perf_melted, p9.aes(x='length_bin', y='SMAPE', color='Method', group='Method'))
        + p9.geom_line(size=1.5)
        + p9.geom_point(size=3)
        + p9.labs(x='Series Length Bin', y='Average SMAPE', 
                 title='Performance vs Series Length')
        + THEME)

plot.save('test2.pdf', width=8, height=6)

# Determine winner for each series
df['winner'] = df[['MetaARIMA', 'AutoETS']].idxmin(axis=1)

# Calculate win rates
win_rates = df.groupby(['length_bin', 'winner']).size().unstack(fill_value=0)
win_rates_pct = win_rates.div(win_rates.sum(axis=1), axis=0) * 100

# Plot stacked bar chart
win_rates_melted = win_rates_pct.reset_index().melt(id_vars='length_bin',
                                                    var_name='Method',
                                                    value_name='Win Rate %')

plot = (p9.ggplot(win_rates_melted, p9.aes(x='length_bin', y='Win Rate %', fill='Method'))
        + p9.geom_bar(stat='identity', position='stack')
        + p9.labs(x='Series Length Bin', y='Win Rate (%)',
                 title='Method Win Rates by Series Length')
        + THEME)

plot.save('test3.pdf', width=8, height=6)
