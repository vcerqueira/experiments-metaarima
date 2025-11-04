from pprint import pprint

import pandas as pd
import plotnine as p9

from src.utils import THEME

PLOT_NAME = 'assets/results/plots/transfer_scores.pdf'

results_df = pd.read_csv('assets/results/sensitivity/transfer,M4,Monthly.csv')

avg_scores = results_df.mean(numeric_only=True).reset_index()
avg_scores = results_df.groupby('dataset').mean(numeric_only=True).reset_index()
avg_scores.columns = ['variable', 'SMAPE']

plot = p9.ggplot(avg_scores, p9.aes(**{'x': 'variable', 'y': 'SMAPE'})) + \
       THEME + \
       p9.theme(plot_margin=0.015,
                axis_text_y=p9.element_text(size=12),
                axis_text_x=p9.element_text(size=12, angle=30),
                legend_title=p9.element_blank(),
                legend_position=None,
                strip_text=p9.element_text(size=13)) + \
       p9.geom_bar(stat='identity', fill='teal') + \
       p9.labs(x='')

plot.save(PLOT_NAME, width=11, height=4.5)
