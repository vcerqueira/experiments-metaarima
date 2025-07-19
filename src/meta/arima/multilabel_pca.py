import warnings

from sklearn.preprocessing import StandardScaler
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

    def __init__(self, mod, n_components=50, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

        self.regressor = mod

        self.pca = None
        self.feature_scaler_ = StandardScaler()
        self.n_features_in_ = None
        self.n_outputs_ = None

        self.is_fit = False

    def fit(self, X, y):
        """
        todo create sphinx-compatible docstrings
        """
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]

        X_scaled = self.feature_scaler_.fit_transform(X)

        self.n_components = min(self.n_components, y.shape[1])

        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        Z = self.pca.fit_transform(y)

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

        return Y_pred

    def predict_proba(self, X):
        """Get prediction probabilities (same as predict for regression), just normalized."""

        # preds = self.predict(X)

        # min_vals = preds.min(axis=1, keepdims=True)
        # max_vals = preds.max(axis=1, keepdims=True)
        # preds_norm = 1 - (preds - min_vals) / (max_vals - min_vals)

        # return preds_norm
        return self.predict(X)
