"""
1. dist boxplot of error across all time series
    include mean

    mostrar que tem melhores resultados wrt to outros metodos para configurar arima e metodos sota
    talvez dizer que em media n e tao notorio


2. tabela1 com resultados médios por dataset + avg rank
    tabela2 com resultados médianos por dataset + avg rank


sens analysis
3. barplot ou tabela dos resultados das variantes

4. lineplot com resultados das outras sens analysis
    n trials
    quantile
    lambda

5. tabela com resultados por dataset na analise TL


"""
import numpy as np
import pandas as pd
import plotnine as p9

from src.utils import to_latex_tab, THEME, read_results

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

df = read_results()

df_mt = df.drop(columns=['Dataset']).melt()

avg_score = df_mt.groupby('variable', observed=False)['value'].mean().reset_index()

aes_ = {'x': 1, 'y': 'np.log(value+1)'}

df_mt_blp2 = df_mt.query('value<0.2')

plot = p9.ggplot(df_mt_blp2, p9.aes(**aes_)) + \
       p9.theme_minimal(base_family='Palatino', base_size=12) + \
       p9.theme(plot_margin=0.015,
                axis_text_y=p9.element_text(size=12),
                axis_text_x=p9.element_blank(),
                # axis_line_x=p9.element_blank(),
                legend_title=p9.element_blank(),
                legend_position=None) + \
       p9.facet_grid(f'. ~variable') + \
       p9.geom_violin(  # fill='#58a63e',
           # color='variable',
           show_legend=False) + \
       p9.geom_hline(data=avg_score,
                     mapping=p9.aes(yintercept='value'),
                     colour='red',
                     size=1) + \
       p9.xlab('') + \
       p9.ylab('MASE')

plot.save('img.pdf', width=12, height=5)

avg_by_ds = df.groupby(['Dataset']).mean(numeric_only=True).round(4)
avg_by_ds.loc['Avg.', :] = avg_by_ds.mean().values
avg_by_ds.loc['Avg. Rank', :] = avg_by_ds.rank(axis=1).mean().round(2).values

med_by_ds = df.groupby(['Dataset']).median(numeric_only=True).round(4)
med_by_ds.loc['Avg.', :] = med_by_ds.mean().values
med_by_ds.loc['Avg. Rank', :] = med_by_ds.rank(axis=1).mean().round(2).values

avg_by_ds_tab = to_latex_tab(avg_by_ds, 4, rotate_cols=True)
print(avg_by_ds_tab)

med_by_ds_tab = to_latex_tab(med_by_ds, 4, rotate_cols=True)
print(med_by_ds_tab)
