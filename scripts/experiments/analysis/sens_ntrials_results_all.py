import os
import glob

import pandas as pd
import plotnine as p9

from src.utils import THEME

PLOT_NAME = 'assets/results/plots/ntrials_scores.pdf'
SENSITIVITY_DIR = 'assets/results/sensitivity'


def load_ntrials_results():
    """Load ntrials sensitivity results and compute per-dataset ranks."""
    pattern = os.path.join(SENSITIVITY_DIR, '*ntrials*csv')
    paths = sorted(glob.glob(pattern))

    all_meta_rows = []

    for path in paths:
        results_df = pd.read_csv(path)
        avg_scores = results_df.mean(numeric_only=True)

        meta_mask = avg_scores.index.str.contains('MetaARIMA')
        if not meta_mask.any():
            continue

        ntrials = (
            avg_scores.index[meta_mask]
            .str.extract(r'\((.*?)\)')[0]
            .astype(int)
        )

        dataset_name = os.path.basename(path)
        dataset_name = dataset_name.replace('ntrials,', '').replace('.csv', '')

        df_meta = pd.DataFrame(
            {
                'ntrials': ntrials.values,
                'MASE': avg_scores[meta_mask].values,
                'dataset': dataset_name,
            }
        )

        df_meta['rank'] = df_meta['MASE'].rank(ascending=True, method='average')

        all_meta_rows.append(df_meta)

    df_meta_all = pd.concat(all_meta_rows, ignore_index=True)

    return df_meta_all


df_meta_all = load_ntrials_results()

avg_ranks = (
    df_meta_all.groupby('ntrials', as_index=False)['rank']
    .mean()
    .rename(columns={'rank': 'avg_rank'})
    .sort_values('ntrials')
)


plot = (
    p9.ggplot(avg_ranks, p9.aes(x='ntrials', y='avg_rank'))
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
#     mapping=p9.aes(x='ntrials', y='avg_rank', group=1),
#     method='loess',
#     se=False,
#     colour='orangered',
#     size=1.2,
# )

plot = (
    plot
    + p9.scale_x_continuous(breaks=range(0, int(avg_ranks['ntrials'].max()) + 10, 10))
    + p9.xlab('Number of trials')
    + p9.ylab('Average MASE rank')
)

plot.save(PLOT_NAME, width=12, height=4)
