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
avg_by_ds.loc['Avg.',:] = avg_by_ds.mean().values
avg_by_ds.loc['Avg. Rank',:] = avg_by_ds.rank(axis=1).mean().round(2).values

med_by_ds = df.groupby(['Dataset']).median(numeric_only=True).round(4)
med_by_ds.loc['Avg.',:] = med_by_ds.mean().values
med_by_ds.loc['Avg. Rank',:] = med_by_ds.rank(axis=1).mean().round(2).values



df.groupby(['Dataset']).mean(numeric_only=True).mean()
df.groupby('Dataset').apply(lambda x: x.rank(axis=1)).mean()
df.groupby(['Dataset']).mean(numeric_only=True).median()

avg_perf = perf_by_all.reset_index().groupby('model').mean(numeric_only=True)
avg_rank = perf_by_all.rank(axis=1).reset_index(level='model').groupby('model').mean(numeric_only=True).round(2)
avg_rank_ds = perf_by_all.rank(axis=1).reset_index(level='ds').groupby('ds').mean(numeric_only=True).round(2)
avg_perf_ds = perf_by_all.reset_index().groupby('ds').mean(numeric_only=True)

# perf_by_mod = df.groupby(['model']).mean(numeric_only=True)
# avg_score = perf_by_mod.mean().values
# avg_rank = perf_by_all.rank(axis=1).mean().round(2).values

# perf_by_mod.loc[('All', 'Average'), :] = avg_score
# perf_by_all.loc[('All', 'Avg. Rank'), :] = avg_rank
perf_by_all_ext.loc[('All', 'Effectiveness'), :] = effectiveness.round(2)

#
df = perf_by_all_ext.copy()

df = df.round(4)

annotated_res_list = []
for i, r in df.iterrows():
    print(i)
    top_2 = r.sort_values().unique()[:2]
    if len(top_2) < 2:
        raise ValueError('only one score')

    best1 = r[r == top_2[0]].values[0]
    best2 = r[r == top_2[1]].values[0]

    r[r == top_2[0]] = f'\\textbf{{{best1}}}'
    r[r == top_2[1]] = f'\\underline{{{best2}}}'

    star_models = []
    if i[1] not in ['Avg.', 'Avg. Rank', 'Effectiveness']:
        wilc_i = wilc_binary.loc[i]

        for m_ in r.index:
            if m_ == 'Original':
                continue

            if df.loc[i][m_] < df.loc[i]['Original']:
                # if wilc_i[m_]:
                #     r[m_] += '??'
                star_models.append(m_)

    # star_models = ['Scaling', 'MBB']
    # for model_key in star_models:
    #     if model_key in r.index:  # Ensure the key exists
    # r[model_key] = str(r[model_key]) + '*'

    if len(star_models) > 0:
        r[star_models] = [f'{x}*' for x in r[star_models]]

    r = r.astype(str)
    print(r)

    annotated_res_list.append(r)

annotated_res = pd.DataFrame(annotated_res_list).astype(str)

annotated_res.index = pd.MultiIndex.from_tuples(
    [(f'\\rotatebox{{90}}{{{x[0]}}}', x[1]) for x in annotated_res.index]
)
annotated_res.columns = [f'\\rotatebox{{60}}{{{x}}}' for x in annotated_res.columns]

text_tab = annotated_res.to_latex(caption='CAPTION', label='tab:scores_by_ds')

#

tex_tab = to_latex_tab(text_tab, 4, rotate_cols=True)
print(tex_tab)
