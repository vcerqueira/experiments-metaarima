from sklearn.metrics import mutual_info_score
import numpy as np
import pandas as pd


def compute_mutual_info_matrix(df, n_bins=30):
    """
    Compute mutual information matrix for all numeric columns in a DataFrame.
    Returns a symmetric matrix of mutual information scores.
    """
    n_vars = df.shape[1]
    mi_matrix = np.zeros((n_vars, n_vars))
    cols = df.columns

    for i in range(n_vars):
        for j in range(i, n_vars):
            c1 = df[cols[i]]
            c2 = df[cols[j]]

            # Create histograms of the data
            c1_hist = np.histogram(c1, bins=n_bins)[0]
            c2_hist = np.histogram(c2, bins=n_bins)[0]

            # Compute joint histogram
            hist_2d = np.histogram2d(c1, c2, bins=n_bins)[0]

            # Compute mutual information using the histograms
            mi = mutual_info_score(None, None,
                                   contingency=hist_2d)

            # Matrix is symmetric
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return pd.DataFrame(mi_matrix, index=cols, columns=cols)
