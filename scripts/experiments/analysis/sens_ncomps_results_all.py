import os
import glob

import pandas as pd
import plotnine as p9

from src.utils import THEME

PLOT_NAME = 'assets/results/plots/ncomps_scores.pdf'
SENSITIVITY_DIR = 'assets/results/sensitivity'


def load_ncomps_results():
    """Load ncomps sensitivity results and compute per-dataset ranks."""
    pattern = os.path.join(SENSITIVITY_DIR, '*ncomps*csv')
    paths = sorted(glob.glob(pattern))

    all_meta_rows = []

    for path in paths:
        results_df = pd.read_csv(path)
        avg_scores = results_df.mean(numeric_only=True)

        meta_mask = avg_scores.index.str.contains('MetaARIMA')
        if not meta_mask.any():
            continue

        ncomps = (
            avg_scores.index[meta_mask]
            .str.extract(r'\((.*?)\)')[0]
            .astype(int)
        )

        dataset_name = os.path.basename(path)
        dataset_name = dataset_name.replace('ncomps,', '').replace('.csv', '')

        df_meta = pd.DataFrame(
            {
                'ncomps': ncomps.values,
                'MASE': avg_scores[meta_mask].values,
                'dataset': dataset_name,
            }
        )

        df_meta['rank'] = df_meta['MASE'].rank(ascending=True, method='average')

        all_meta_rows.append(df_meta)

    df_meta_all = pd.concat(all_meta_rows, ignore_index=True)

    return df_meta_all


df_meta_all = load_ncomps_results()

avg_ranks = (
    df_meta_all.groupby('ncomps', as_index=False)['rank']
    .mean()
    .rename(columns={'rank': 'avg_rank'})
    .sort_values('ncomps')
)

plot = (
        p9.ggplot(avg_ranks, p9.aes(x='ncomps', y='avg_rank'))
        + THEME
        + p9.theme(
    plot_margin=0.015,
    axis_text_y=p9.element_text(size=12),
    axis_text_x=p9.element_text(size=12),
    legend_title=p9.element_blank(),
    legend_position='right',
    strip_text=p9.element_text(size=13),
)
        + p9.geom_point()
        + p9.geom_line()
)

# plot = plot + p9.geom_smooth(
#     mapping=p9.aes(x='ncomps', y='avg_rank', group=1),
#     method='loess',
#     se=False,
#     colour='orangered',
#     size=1.2,
# )

plot = plot + p9.xlab('Number of principal components') + p9.ylab('Average MASE rank')

plot.save(PLOT_NAME, width=12, height=4)
