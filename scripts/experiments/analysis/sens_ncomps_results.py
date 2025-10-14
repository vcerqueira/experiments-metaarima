import pandas as pd
import plotnine as p9

from src.utils import THEME

PLOT_NAME = 'assets/results/plots/ncomps_scores.pdf'

results_df = pd.read_csv('assets/results/sensitivity/ncomps,M3,Monthly.csv', index_col='ncomps')

meta_arima_mask = results_df.index.str.contains('MetaARIMA')
df_meta = pd.DataFrame({
    'ncomps': results_df.index[meta_arima_mask].str.extract(r'\((.*?)\)')[0].astype(float),
    'SMAPE': results_df['avg'].values
})

plot = p9.ggplot(df_meta, p9.aes(**{'x': 'ncomps', 'y': 'SMAPE'})) + \
       THEME + \
       p9.theme(plot_margin=0.015,
                axis_text_y=p9.element_text(size=12),
                axis_text_x=p9.element_text(size=12),
                legend_title=p9.element_blank(),
                legend_position=None,
                strip_text=p9.element_text(size=13)) + \
       p9.geom_point() + \
       p9.geom_line(group=1) + \
       p9.xlab('Number of PC') + \
       p9.ylab('SMAPE')

plot.save(PLOT_NAME, width=12, height=3.5)
