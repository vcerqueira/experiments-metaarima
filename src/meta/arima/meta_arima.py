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

        if self.meta_regression:
            self.meta_model.fit(X, Y)
        else:
            self.meta_model.fit(X, y)

        self.is_fit = True

    def meta_predict(self, X):
        assert self.is_fit

        if self.meta_regression:
            meta_preds = self.meta_model.predict(X)
        else:
            meta_preds = self.meta_model.predict_proba(X)

        if isinstance(meta_preds, list):
            meta_preds = np.asarray([x[:, 1] for x in meta_preds]).T

        if self.meta_regression:
            min_vals = meta_preds.min(axis=1, keepdims=True)
            max_vals = meta_preds.max(axis=1, keepdims=True)
            meta_preds = 1 - (meta_preds - min_vals) / (max_vals - min_vals)

        if self.use_mmr:
            meta_preds = [pd.Series(x, index=self.model_names) for x in meta_preds]

            meta_preds_list = []
            for i, meta_pred in enumerate(meta_preds):
                selected_indices = self._mmr_selection(probs=meta_pred)

                mod_list = meta_pred.index[selected_indices].tolist()

                meta_preds_list.append(mod_list)
        else:
            preds = pd.DataFrame(meta_preds, columns=self.model_names)

            meta_preds_list = preds.apply(lambda x: x.sort_values().index[:self.n_trials].tolist(),
                                          axis=1).values.tolist()

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

    def _mmr_selection(self, probs: pd.Series):
        """ Re-rank configurations based on maximal marginal relevance

        Re-rank configurations based on maximal marginal relevance to balance accuracy and diversity
        (low correlation with already selected configurations).

        Parameters:
        -----------
        probs : pd.Series, shape (n_configurations,)
            The predicted probabilities of each configuration being in the top percentile (quantile_thr).

        Returns:
        --------
        selected_indices : list
            Indices of the selected configurations.
        """
        n_configs = len(probs)

        selected_indices = []

        remaining_indices = set(range(n_configs))
        best_idx = np.argmax(probs)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        while len(selected_indices) < self.n_trials and remaining_indices:
            max_mmr_score = -float('inf')
            next_best_idx = None

            for idx in remaining_indices:
                # max corr with already selected configurations
                max_corr = max([self.corr_mat.iloc[idx, sel_idx] for sel_idx in selected_indices])

                # MMR
                # High lambda: -> high probability
                # Low lambda: -> low correlation
                mmr_score = self.mmr_lambda * probs.iloc[idx] - (1 - self.mmr_lambda) * max_corr

                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    next_best_idx = idx

            if next_best_idx is not None:
                selected_indices.append(next_best_idx)
                remaining_indices.remove(next_best_idx)
            else:
                break

        return selected_indices

    @classmethod
    def from_model(cls, n_trials: int):
        # todo implement serialiation

        pass

    @staticmethod
    def _check_params(quantile_thr, mmr_lambda):
        assert quantile_thr > 0
        assert quantile_thr < 1
        assert mmr_lambda >= 0
        assert mmr_lambda <= 1
