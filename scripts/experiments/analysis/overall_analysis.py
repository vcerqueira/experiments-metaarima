import pandas as pd
import plotnine as p9

from src.utils import to_latex_tab, THEME, read_results

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

df = read_results()

perf_by_ds = df.groupby(['Dataset']).mean(numeric_only=True)

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
