import warnings
import copy
from typing import List

import numpy as np
import pandas as pd

from src.meta.arima.multilabel_pca import MultiLabelPCARegressor
from src.meta.arima._base import (MetaARIMAUtils,
                                 _MetaARIMABase,
                                 _HalvingMetaARIMABase,
                                 _MetaARIMABaseMC)

warnings.filterwarnings(action='ignore')


class MetaARIMA:
    """Optimized version of MetaARIMA with improved meta_predict and _mmr_selection methods."""
    
    BASE_OPTIM_METHODS = ['complete', 'mc', 'halving']

    def __init__(self,
                 model,
                 freq: str,
                 season_length: int,
                 n_trials: int,
                 meta_regression: bool = False,
                 target_pca: bool = True,
                 base_optim: str = 'halving',
                 quantile_thr: float = 0.05,
                 mmr_lambda: float = 0.75,
                 use_mmr: bool = True):

        self.n_trials = n_trials
        self.quantile_thr = quantile_thr
        self.model_names = None
        self.freq = freq
        self.season_length = season_length
        self.model = None
        self.corr_mat = None
        self.corr_mat_values = None  # Numpy array version for faster access
        self.target_pca = target_pca
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.base_optim = base_optim
        self.meta_regression = meta_regression
        self.selected_config = ''
        if self.target_pca:
            self.meta_model = copy.deepcopy(MultiLabelPCARegressor(mod=model))
        else:
            self.meta_model = copy.deepcopy(model)

        self.is_fit: bool = False

        assert self.base_optim in self.BASE_OPTIM_METHODS

    def meta_fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        """
        
        :param X: feature set
        :param Y: error scores
        :return:
        """
        self.model_names = Y.columns.tolist()

        y = Y.apply(lambda x: (x <= x.quantile(self.quantile_thr)).astype(int), axis=1)
        if self.use_mmr:
            self.corr_mat = Y.corr()
            # Store as numpy array for faster access
            self.corr_mat_values = self.corr_mat.values

        if self.meta_regression:
            self.meta_model.fit(X, Y)
        else:
            self.meta_model.fit(X, y)

        self.is_fit = True

    def meta_predict(self, X):
        """Optimized meta_predict with vectorized operations."""
        assert self.is_fit

        if self.meta_regression:
            meta_preds = self.meta_model.predict(X)
        else:
            meta_preds = self.meta_model.predict_proba(X)

        if isinstance(meta_preds, list):
            meta_preds = np.asarray([x[:, 1] for x in meta_preds]).T

        if self.meta_regression:
            # Vectorized min-max normalization
            min_vals = meta_preds.min(axis=1, keepdims=True)
            max_vals = meta_preds.max(axis=1, keepdims=True)
            meta_preds = 1 - (meta_preds - min_vals) / (max_vals - min_vals)

        if self.use_mmr:
            # Pre-allocate list for better performance
            meta_preds_list = []
            
            for i in range(meta_preds.shape[0]):
                meta_pred = meta_preds[i]
                selected_indices = self._mmr_selection(meta_pred)
                mod_list = [self.model_names[idx] for idx in selected_indices]
                meta_preds_list.append(mod_list)
        else:
            # Vectorized top-k selection using argpartition for better performance
            # argpartition is O(n) vs argsort which is O(n log n)
            meta_preds_list = []
            for i in range(meta_preds.shape[0]):
                # Get indices of top n_trials items (smallest values = best predictions)
                if self.n_trials < len(self.model_names):
                    # Use argpartition for partial sorting - faster for small k
                    top_indices = np.argpartition(meta_preds[i], self.n_trials)[:self.n_trials]
                    # Sort only the top k elements
                    top_indices_sorted = top_indices[np.argsort(meta_preds[i][top_indices])]
                else:
                    # If we need all or most elements, regular argsort is fine
                    top_indices_sorted = np.argsort(meta_preds[i])[:self.n_trials]
                
                mod_list = [self.model_names[idx] for idx in top_indices_sorted]
                meta_preds_list.append(mod_list)

        return meta_preds_list

    def fit(self, df: pd.DataFrame, config_space: List[str]):
        assert self.is_fit

        base_params = {'config_space': config_space,
                       'freq': self.freq,
                       'season_length': self.season_length}

        if self.base_optim == 'complete':
            self.model = _MetaARIMABase(**base_params)
        elif self.base_optim == 'mc':
            # todo hardcoded params1...
            self.model = _MetaARIMABaseMC(**base_params, n_trials=10, trial_n_obs=0.4)
        elif self.base_optim == 'halving':
            # todo hardcoded params2...
            self.model = _HalvingMetaARIMABase(**base_params, eta=2, init_resource_factor=4, resource_factor=2)
        else:
            raise ValueError(f'Unknown base optimizer: {self.base_optim}')

        self.model.fit(df)
        self.selected_config = MetaARIMAUtils.get_model_order(self.model.sf.fitted_[0][0].model_,
                                                              as_alias=True,
                                                              alias_freq=self.season_length)

    def _mmr_selection(self, probs: np.ndarray):
        """Optimized MMR selection using vectorized operations.
        
        Re-rank configurations based on maximal marginal relevance to balance 
        accuracy and diversity (low correlation with already selected configurations).

        Parameters:
        -----------
        probs : np.ndarray, shape (n_configurations,)
            The predicted probabilities of each configuration being in the top percentile.

        Returns:
        --------
        selected_indices : list
            Indices of the selected configurations.
        """
        n_configs = len(probs)
        
        # Pre-allocate arrays
        selected_indices = []
        remaining_mask = np.ones(n_configs, dtype=bool)
        
        # Select first best index
        best_idx = np.argmax(probs)
        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False
        
        # Pre-compute lambda-weighted probabilities (doesn't change)
        lambda_probs = self.mmr_lambda * probs
        one_minus_lambda = 1 - self.mmr_lambda
        
        # Main selection loop
        while len(selected_indices) < self.n_trials and remaining_mask.any():
            # Get remaining indices
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_indices) == 0:
                break
            
            # Vectorized max correlation calculation
            # Extract correlations for remaining configs with selected configs
            # Shape: (n_remaining, n_selected)
            corr_matrix = self.corr_mat_values[np.ix_(remaining_indices, selected_indices)]
            
            # Get max correlation for each remaining config
            max_corrs = np.max(corr_matrix, axis=1)
            
            # Vectorized MMR score calculation
            mmr_scores = lambda_probs[remaining_indices] - one_minus_lambda * max_corrs
            
            # Find best scoring index
            best_relative_idx = np.argmax(mmr_scores)
            next_best_idx = remaining_indices[best_relative_idx]
            
            selected_indices.append(next_best_idx)
            remaining_mask[next_best_idx] = False
        
        return selected_indices

    @classmethod
    def from_model(cls, n_trials: int):
        # todo implement serialization
        pass

    @staticmethod
    def _check_params(quantile_thr, mmr_lambda):
        assert quantile_thr > 0
        assert quantile_thr < 1
        assert mmr_lambda >= 0
        assert mmr_lambda <= 1

