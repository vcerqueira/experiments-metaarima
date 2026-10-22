import os
import glob

import pandas as pd
import plotnine as p9

from src.utils import THEME

PLOT_NAME = 'assets/results/plots/ncomps_scores.pdf'
SENSITIVITY_DIR = 'assets/results/sensitivity'


def load_ncomps_results():
    """Load ncomps sensitivity results and average the MASE rank of each series."""
    pattern = os.path.join(SENSITIVITY_DIR, '*ncomps*csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matched {pattern}. Run scripts/experiments/main/sensitivity/sensitivity_ncomps.py first."
        )

    all_meta_rows = []

    for path in paths:
        results_df = pd.read_csv(path)
        meta_cols = results_df.columns[results_df.columns.str.contains('MetaARIMA')]
        if len(meta_cols) == 0:
            continue

        ncomps = meta_cols.str.extract(r'\((.*?)\)')[0].astype(int)
        series_ranks = results_df[meta_cols].rank(axis=1, ascending=True, method='average')

        dataset_name = os.path.basename(path)
        dataset_name = dataset_name.replace('ncomps,', '').replace('.csv', '')

        df_meta = pd.DataFrame(
            {
                'ncomps': ncomps.values,
                'MASE': results_df[meta_cols].mean().values,
                'dataset': dataset_name,
                'rank': series_ranks.mean().values,
            }
        )

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

os.makedirs('assets/results/plots', exist_ok=True)
plot.save(PLOT_NAME, width=12, height=4)
