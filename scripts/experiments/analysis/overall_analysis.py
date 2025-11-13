import warnings

import numpy as np
import pandas as pd
import plotnine as p9

from src.utils import to_latex_tab, THEME, read_results

warnings.simplefilter(action='ignore', category=FutureWarning)

TRUNCATE_DIST = True

df = read_results()
df_mt = df.drop(columns=['Dataset', 'Frequency', 'unique_id']).melt()

avg_score = df_mt.groupby('variable', observed=False)['value'].mean().reset_index()
med_score = df_mt.groupby('variable', observed=False)['value'].median().reset_index()
sorted_vars = avg_score.sort_values('value', ascending=True)['variable'].tolist()
df_mt['variable'] = pd.Categorical(df_mt['variable'], categories=sorted_vars)
avg_score['variable'] = pd.Categorical(avg_score['variable'], categories=sorted_vars)
med_score['variable'] = pd.Categorical(med_score['variable'], categories=sorted_vars)

# aes_ = {'x': 1, 'y': 'np.log(value+1)'}
aes_ = {'x': 1, 'y': 'value'}

df_mt_blp2 = df_mt.query('value<4')
df_mt_blp2['variable'] = pd.Categorical(df_mt_blp2['variable'], categories=sorted_vars)

if TRUNCATE_DIST:
    plot = p9.ggplot(df_mt_blp2, p9.aes(**aes_))
    plot_name = 'assets/results/plots/mase_distr_violins_trunc.pdf'
else:
    plot = p9.ggplot(df_mt, p9.aes(**aes_))
    plot_name = 'assets/results/plots/mase_distr_violins.pdf'

plot = plot + \
       THEME + \
       p9.theme(plot_margin=0.015,
                axis_text_y=p9.element_text(size=12),
                axis_text_x=p9.element_blank(),
                axis_line_x=p9.element_blank(),
                axis_ticks_x=p9.element_blank(),
                legend_title=p9.element_blank(),
                legend_position=None,
                strip_text=p9.element_text(size=13)) + \
       p9.facet_grid('. ~variable') + \
       p9.geom_violin(show_legend=False) + \
       p9.geom_hline(data=avg_score,
                     mapping=p9.aes(yintercept='value'),
                     colour='orangered',
                     size=1.3) + \
       p9.geom_hline(data=med_score,
                     mapping=p9.aes(yintercept='value'),
                     colour='blue',
                     size=1.3) + \
       p9.xlab('') + \
       p9.ylab('MASE')

plot.save(plot_name, width=12, height=5)

avg_by_ds = df.groupby(['Dataset']).mean(numeric_only=True).round(4)
# avg_by_ds.loc['Avg.', :] = avg_by_ds.mean().values
avg_by_ds.loc['Avg. Rank', :] = avg_by_ds.rank(axis=1).mean().round(2).values

med_by_ds = df.groupby(['Dataset']).median(numeric_only=True).round(4)
# med_by_ds.loc['Avg.', :] = med_by_ds.mean().values
med_by_ds.loc['Avg. Rank', :] = med_by_ds.rank(axis=1).mean().round(2).values

avg_by_ds_tab = to_latex_tab(avg_by_ds, 4, rotate_cols=True)
print(avg_by_ds_tab)

med_by_ds_tab = to_latex_tab(med_by_ds, 4, rotate_cols=True)
print(med_by_ds_tab)

# df[sorted_vars].apply(lambda x: x.rank()==1.0, axis=1).astype(int).mean()
