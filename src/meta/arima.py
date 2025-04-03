import warnings
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsforecast.models import ARIMA
from statsforecast import StatsForecast

warnings.filterwarnings(action='ignore')


class MetaARIMABase:

    def __init__(self, config_space: List[str], season_length: int, freq: str):
        self.config_space = config_space
        self.freq = freq
        self.season_length = season_length
        self.models = MetaARIMAUtils.get_models_sf(season_length=self.season_length, alias_list=self.config_space)
        self.nmodels = len(self.models)
        self.sf = StatsForecast(models=self.models, freq=self.freq)

        self.alias = 'MetaARIMA'

    def fit(self, df: pd.DataFrame):
        self.sf.fit(df=df)

        aicc_ = [self.sf.fitted_[0][i].model_['aicc'] for i in range(self.nmodels)]

        assert len(aicc_) == self.nmodels

        best_idx = np.array(aicc_).argmin()

        self.sf.fitted_ = np.array([[self.sf.fitted_[0][best_idx]]])
        self.sf.fitted_[0][0].alias = self.alias

    def predict(self, h: int):
        return self.sf.predict(h=h)


class MetaARIMABaseMC:

    def __init__(self,
                 config_space: List[str],
                 n_trials: int,
                 season_length: int,
                 freq: str):

        self.config_space = config_space
        self.freq = freq
        self.season_length = season_length
        self.models = MetaARIMAUtils.get_models_sf(season_length=self.season_length, alias_list=self.config_space)
        self.nmodels = len(self.models)
        self.sf = StatsForecast(models=self.models, freq=self.freq)
        self.n_trials = n_trials

        self.alias = 'MetaARIMA'

    def fit(self, df: pd.DataFrame):
        # do this for each trial
        self.sf.fit(df=df)

        aicc_ = [self.sf.fitted_[0][i].model_['aicc'] for i in range(self.nmodels)]

        assert len(aicc_) == self.nmodels

        # select the one with lowest average aicc
        best_idx = np.array(aicc_).argmin()

        # THEN, refit (pass to MetaARIMABase??) best one with all data

        self.sf.fitted_ = np.array([[self.sf.fitted_[0][best_idx]]])
        self.sf.fitted_[0][0].alias = self.alias

    def predict(self, h: int):
        return self.sf.predict(h=h)



class MetaARIMAUtils:
    ORDER_MAX = {'AR': 2, 'I': 1, 'MA': 2, 'S_AR': 1, 'S_I': 1, 'S_MA': 1, }

    @staticmethod
    def get_model_order(mod, as_alias: bool = False, alias_freq: int = 1):
        order = tuple(mod['arma'][i] for i in [0, 5, 1, 2, 6, 3, 4])

        ord = pd.Series(order, index=['AR', 'I', 'MA', 'S_AR', 'S_I', 'S_MA', 'm'])

        if as_alias:
            alias = f'ARIMA({ord[0]},{ord[1]},{ord[2]})({ord[3]},{ord[4]},{ord[5]})[{alias_freq}]'
            return alias

        return ord

    @classmethod
    def get_models_sf(cls,
                      season_length: int,
                      return_names: bool = False,
                      max_config: Optional[Dict] = None,
                      alias_list: Optional[List[str]] = None):

        if max_config is None:
            max_config_ = cls.ORDER_MAX
        else:
            max_config_ = max_config

        models = []
        for ar in range(max_config_['AR'] + 1):
            for i in range(max_config_['I'] + 1):
                for ma in range(max_config_['MA'] + 1):
                    for s_ar in range(max_config_['S_AR'] + 1):
                        for s_i in range(max_config_['S_I'] + 1):
                            for s_ma in range(max_config_['S_MA'] + 1):
                                models.append(
                                    ARIMA(order=(ar, i, ma),
                                          season_length=season_length,
                                          seasonal_order=(s_ar, s_i, s_ma),
                                          alias=f'ARIMA({ar},{i},{ma})({s_ar},{s_i},{s_ma})[{season_length}]')
                                )

        if return_names:
            model_names = [x.alias for x in models]
            return model_names

        if alias_list is not None:
            models = [x for x in models if x.alias in alias_list]

        return models

    @classmethod
    def model_summary(cls, model):

        coefs = {f'coef_{k}': model['coef'][k] for k in model['coef']}

        try:
            var_coef_avg = model['var_coef'].mean()
        except AttributeError:
            var_coef_avg = np.nan

        goodness_fit = {
            'var_coef_mean': var_coef_avg,
            'aic': model['aic'],
            'aicc': model['aicc'],
            'bic': model['bic'],
            'loglik': model['loglik'],
        }

        resid_tests = cls.residual_testing(model['residuals'])

        mod_summary = {**coefs, **goodness_fit, **resid_tests}

        return mod_summary

    @staticmethod
    def residual_testing(residuals):
        """
        Return p-values for time series residual diagnostic tests

        Parameters:
        residuals: array-like, the model residuals to test

        Returns:
        dict: Dictionary containing p-values for the following hypothesis tests:
            zero_mean: Tests H0: mean = 0 vs H1: mean ≠ 0
            constant_variance: Tests H0: variance is constant vs H1: variance changes over time
            normality: Tests H0: residuals are normally distributed vs H1: residuals are not normal
            no_autocorrelation: Tests H0: no autocorrelation at lag 1 vs H1: autocorrelation exists at lag 1

        For all tests, p > 0.05 indicates failure to reject H0 (desired outcome for residuals)

        """
        # Mean test
        mean_test = stats.ttest_1samp(residuals, 0).pvalue

        # var test
        squared_resid = residuals ** 2
        trend = np.arange(len(residuals))
        bp_test = stats.linregress(trend, squared_resid).pvalue

        # normality
        jb_test = jarque_bera(residuals)[1]

        # acf test (on lag 1)
        lb_test = acorr_ljungbox(residuals, lags=[1]).lb_pvalue.values[0]

        pvals = {
            'zero_mean': mean_test,
            'constant_variance': bp_test,
            'normality': jb_test,
            'no_autocorrelation': lb_test,
        }

        return pvals


class MetaARIMA:

    def __init__(self,
                 model,
                 freq: str,
                 season_length: int,
                 n_trials: int,
                 quantile_thr: float = 0.05,
                 mmr_lambda: float = 0.75,
                 use_mmr: bool = False):

        self.meta_model = model
        self.n_trials = n_trials
        self.quantile_thr = quantile_thr
        self.model_names = None
        self.freq = freq
        self.season_length = season_length
        self.model = None
        self.corr_mat = None
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

        self.is_fit: bool = False

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

        self.meta_model.fit(X, y)
        self.is_fit = True

    def meta_predict(self, X):
        assert self.is_fit

        if self.use_mmr:
            meta_preds = self.meta_model.predict_proba(X)
            if isinstance(meta_preds, list):
                meta_preds = np.asarray([x[:, 1] for x in meta_preds]).T

            meta_preds = [pd.Series(x, index=self.model_names) for x in meta_preds]

            meta_preds_list = []
            for i, meta_pred in enumerate(meta_preds):
                selected_indices = self._mmr_selection(probs=meta_pred)

                mod_list = meta_pred.index[selected_indices].tolist()

                meta_preds_list.append(mod_list)
        else:
            meta_preds = self.meta_model.predict_proba(X)
            if isinstance(meta_preds, list):
                meta_preds = np.asarray([x[:, 1] for x in meta_preds]).T

            preds = pd.DataFrame(meta_preds, columns=self.model_names)

            meta_preds_list = preds.apply(lambda x: x.sort_values().index[:self.n_trials].tolist(),
                                          axis=1).values.tolist()

        return meta_preds_list

    def fit(self, df: pd.DataFrame, config_space: List[str]):
        assert self.is_fit

        self.model = MetaARIMABase(config_space=config_space,
                                   freq=self.freq,
                                   season_length=self.season_length)

        self.model.fit(df)

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
