import pandas as pd
import numpy as np


def prepare_ranking_data(df, input_cols, model_cols):
    """
    Prepare data for learning to rank where:
    - Each original row becomes a query/group
    - Models are the items to be ranked within each group
    - Input variables are features shared across models in the group
    - Errors are converted to relevance scores (lower error = higher relevance)

    Args:
        df: Original DataFrame
        input_cols: List of input variable column names
        model_cols: List of model column names (containing errors)

    Returns:
        ranking_df: DataFrame ready for XGBoost ranking
        features: Array of features
        relevance: Array of relevance scores
        groups: Array of group sizes
    """

    # Create expanded DataFrame where each model becomes a row
    rows = []
    for idx, row in df.iterrows():
        # Get input features for this observation
        base_features = row[input_cols].values

        # For each model, create a row
        for model in model_cols:
            error = row[model]

            # Create row with:
            # - group_id (original row index)
            # - all input features
            # - model error
            # - model name (can be encoded if needed)
            rows.append([idx] + list(base_features) + [error, model])

    # Create new DataFrame
    ranking_df = pd.DataFrame(
        rows,
        columns=['group_id'] + input_cols + ['error', 'model_name']
    )

    # Convert errors to relevance scores (normalize and invert)
    # Higher relevance = better rank = lower error
    ranking_df['relevance'] = ranking_df.groupby('group_id')['error'].transform(
        lambda x: 1 - (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 1
    )

    # Prepare final arrays for XGBoost
    features = ranking_df[input_cols].values
    relevance = ranking_df['relevance'].values
    groups = ranking_df.groupby('group_id').size().values

    return ranking_df, features, relevance, groups
