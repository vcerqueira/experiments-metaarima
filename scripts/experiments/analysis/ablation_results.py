import pandas as pd
import plotnine as p9

from src.utils import THEME

PLOT_NAME = 'assets/results/plots/ablation_scores.pdf'

results_df = pd.read_csv('assets/results/sensitivity/ablation,M3,Monthly.csv')
results_df.columns = [
    'MetaARIMA',
    'No PCA',
    'Regr. Chain',
    'No Binarization',
    'No SH',
    'Monte Carlo',
    'Multi-output Regr.',
    'No MMR',
]

avg_scores = results_df.median(numeric_only=True).reset_index()
avg_scores.columns = ['Variant', 'SMAPE']
avg_scores['Variant'] = pd.Categorical(avg_scores['Variant'],
                                       categories=[
                                           'MetaARIMA',
                                           'No PCA',
                                           'No MMR',
                                           'No Binarization',
                                           'No SH',
                                           'Monte Carlo',
                                           'Regr. Chain',
                                           'Multi-output Regr.',
                                       ])

latex_df = avg_scores.copy()
latex_df['SMAPE'] = latex_df['SMAPE'].round(4)
latex_df = latex_df.set_index('Variant').T
latex_df.columns = [f'\\rotatebox{{60}}{{{x}}}' for x in latex_df.columns]
latex_table = latex_df.to_latex(caption='CAPTION', label='tab:ablation_scores')
print(latex_table)

plot = p9.ggplot(avg_scores, p9.aes(**{'x': 'Variant', 'y': 'SMAPE'})) + \
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
