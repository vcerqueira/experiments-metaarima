import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, MultiOutputMixin
from sklearn.utils.validation import check_X_y, check_array

warnings.filterwarnings('ignore')


class MultiLabelPCARegressor(BaseEstimator, MultiOutputMixin):
    """
    PCA Transformation with Custom Multi-output Regressor.

    This implementation combines:
    1. Label space dimensionality reduction via PCA
    2. Multiple regression in the reduced label space (needs to cover multi-targets--the principal components)
    3. Reconstruction back to original label space
    """

    def __init__(self, mod, n_components=100, random_state=1):
        self.n_components = n_components
        self.random_state = random_state

        self.regressor = mod

        self.pca = None
        self.feature_scaler_ = StandardScaler()
        self.target_scaler_ = RobustScaler()
        self.n_features_in_ = None
        self.n_outputs_ = None

        self.is_fit = False
        self.is_meta_regression = False

    def fit(self, X, y, process_y=False):
        """
        todo create sphinx-compatible docstrings
        """
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)

        X_scaled = self.feature_scaler_.fit_transform(X)

        self.n_components = min(self.n_components, y.shape[1])

        if process_y:
            self.is_meta_regression = True

            # this is done for meta-regression

            y_clean = np.where(np.isfinite(y), y, np.nan)
            for j in range(y_clean.shape[1]):
                finites = y_clean[:, j][np.isfinite(y_clean[:, j])]
                fill = np.percentile(finites, 99) if len(finites) > 0 else 0.0
                y_clean[:, j] = np.where(np.isnan(y_clean[:, j]), fill, y_clean[:, j])

            # Winsorize per column to [1st, 99th] percentile to tame extremes
            for j in range(y_clean.shape[1]):
                lo, hi = np.nanpercentile(y_clean[:, j], [1, 99])
                y_clean[:, j] = np.clip(y_clean[:, j], lo, hi)

            y_to_scale = y_clean

            # y_scaled = self.target_scaler_.fit_transform(y_to_scale)

            Z = self.pca.fit_transform(y_to_scale)
        else:
            Z = self.pca.fit_transform(y)

        # Z_clean = np.where(np.isfinite(Z), Z, np.nan)
        # col_max = np.nanmax(Z_clean, axis=0)
        # col_max = np.where(np.isfinite(col_max), col_max, 0.0)
        # for j in range(Z_clean.shape[1]):
        #     Z_clean[:, j] = np.where(np.isnan(Z_clean[:, j]), col_max[j], Z_clean[:, j])

        # self.regressor.fit(X_scaled, Z_clean)
        self.regressor.fit(X_scaled, Z)

        self.is_fit = True

        return self

    def predict(self, X):
        X = check_array(X)

        if not self.is_fit:
            raise ValueError("This model instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments before using this estimator.")

        X_scaled = self.feature_scaler_.transform(X)

        Z_pred = self.regressor.predict(X_scaled)

        Y_pred = self.pca.inverse_transform(Z_pred)


        # if self.is_meta_regression:
        #     Y_pred = self.target_scaler_.inverse_transform(Y_pred)

        return Y_pred

    def predict_proba(self, X):
        """Get prediction probabilities (same as predict for regression), just normalized."""

        # preds = self.predict(X)

        # min_vals = preds.min(axis=1, keepdims=True)
        # max_vals = preds.max(axis=1, keepdims=True)
        # preds_norm = 1 - (preds - min_vals) / (max_vals - min_vals)

        # return preds_norm
        return self.predict(X)
