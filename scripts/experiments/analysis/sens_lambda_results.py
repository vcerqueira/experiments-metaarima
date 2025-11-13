import pandas as pd
import plotnine as p9

from src.utils import THEME

PLOT_NAME = 'assets/results/plots/lambda_scores.pdf'

results_df = pd.read_csv('assets/results/sensitivity/lambda,monash_m3_monthly.csv')

avg_scores = results_df.mean(numeric_only=True)

meta_arima_mask = avg_scores.index.str.contains('MetaARIMA')
df_meta = pd.DataFrame({
    'lambda': avg_scores.index[meta_arima_mask].str.extract(r'\((.*?)\)')[0].astype(float),
    'MASE': avg_scores[meta_arima_mask].values
})
df_meta['lambda'] = df_meta['lambda'].astype(float)

auto_arima_scr = pd.Series({'AutoARIMA': avg_scores[avg_scores.index == 'AutoARIMA'].iloc[0]})
auto_arima_scr = auto_arima_scr.reset_index()
auto_arima_scr.columns = ['AutoARIMA', 'value']

plot = p9.ggplot(df_meta, p9.aes(**{'x': 'lambda', 'y': 'MASE'})) + \
       THEME + \
       p9.theme(plot_margin=0.015,
                axis_text_y=p9.element_text(size=12),
                axis_text_x=p9.element_text(size=12),
                legend_title=p9.element_blank(),
                legend_position=None,
                strip_text=p9.element_text(size=13)) + \
       p9.geom_point() + \
       p9.geom_line(group=1) + \
       p9.geom_hline(data=auto_arima_scr,
                     mapping=p9.aes(yintercept='value'),
                     colour='orangered',
                     size=1.3) + \
       p9.xlab('λ') + \
       p9.ylab('MASE')

plot.save(PLOT_NAME, width=12, height=3.5)
