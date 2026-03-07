import os
import glob

import pandas as pd
import plotnine as p9

from src.utils import THEME

PLOT_NAME = 'assets/results/plots/ablation_scores.pdf'
TABLE_NAME = 'assets/results/plots/ablation_table_all.tex'
SENSITIVITY_DIR = 'assets/results/sensitivity'

COLUMN_RENAME = {
    'Meta-Regr': 'No-Binarization',
    'Reg-Chain': 'Regr. Chain',
    'MO-Regr': 'Regr. MO',
    'MonteCarlo': 'Monte Carlo',
}


def load_ablation_results():
    """Load ablation results from all datasets and compute per-dataset mean MASE and ranks."""
    pattern = os.path.join(SENSITIVITY_DIR, 'ablation,*csv')
    paths = sorted(glob.glob(pattern))

    per_dataset = []
    all_ranks = []

    for path in paths:
        df = pd.read_csv(path)
        df = df.rename(columns=COLUMN_RENAME)
        if 'No-MMR' in df.columns:
            df = df.drop(columns=['No-MMR'])

        # Variants are all numeric columns (exclude unique_id if present)
        variant_cols = [c for c in df.columns if c != 'unique_id' and pd.api.types.is_numeric_dtype(df[c])]
        if not variant_cols:
            continue

        avg = df[variant_cols].median()
        dataset_name = os.path.basename(path).replace('ablation,', '').replace('.csv', '')

        per_dataset.append(avg.rename(dataset_name))

        # Rank within dataset (lower MASE = better = lower rank)
        ranks = avg.rank(ascending=True, method='average')
        ranks = ranks.reset_index()
        ranks.columns = ['Variant', 'rank']
        ranks['dataset'] = dataset_name
        all_ranks.append(ranks)

    if not per_dataset:
        raise RuntimeError('No ablation results found.')

    table_df = pd.concat(per_dataset, axis=1)
    ranks_df = pd.concat(all_ranks, ignore_index=True)

    return table_df, ranks_df


table_df, ranks_df = load_ablation_results()

# Average rank per variant across datasets
avg_rank_series = ranks_df.groupby('Variant')['rank'].mean()
avg_rank = avg_rank_series.reset_index()
avg_rank.columns = ['Variant', 'avg_rank']
avg_rank = avg_rank.sort_values('avg_rank')

# Table: rows = datasets, columns = variants, then add Avg. rank row
table_export = table_df.copy().T  # Transpose so rows = datasets, columns = variants

# Add Avg. rank as a new row at the bottom
table_export.loc['Avg. rank'] = avg_rank_series.reindex(table_export.columns).values

# Save LaTeX table (round MASE to 4 decimals; Avg. rank to 2)
latex_df = table_export.copy()
latex_df = latex_df.round(4)
latex_df.loc['Avg. rank'] = latex_df.loc['Avg. rank'].round(2)
latex_df = latex_df.astype(str)
latex_table = latex_df.to_latex(caption='Ablation study: MASE per dataset (rows) and average rank across variants (bottom row).', label='tab:ablation_scores_all')
print(latex_table)


# Barplot: average rank per variant (lower is better)
avg_rank['Variant'] = pd.Categorical(avg_rank['Variant'], categories=avg_rank['Variant'].tolist())

plot = (
    p9.ggplot(avg_rank, p9.aes(x='Variant', y='avg_rank'))
    + THEME
    + p9.theme(
        plot_margin=0.015,
        axis_text_y=p9.element_text(size=12),
        axis_text_x=p9.element_text(size=12, angle=30),
        legend_title=p9.element_blank(),
        legend_position=None,
        strip_text=p9.element_text(size=13),
    )
    + p9.geom_bar(stat='identity', fill='teal')
    + p9.labs(x='', y='Average rank')
)

plot.save(PLOT_NAME, width=11, height=4.5)
