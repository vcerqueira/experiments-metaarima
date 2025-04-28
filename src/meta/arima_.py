import warnings
from typing import Optional, List, Dict, Union, Tuple
from math import log

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


class HalvingMetaARIMABase(MetaARIMABase):
    def __init__(self,
                 config_space: List[str],
                 season_length: int,
                 freq: str,
                 eta: float = 2,
                 resource_factor: float = 2.5,
                 init_resource_factor: int = 10,
                 min_configs: int = 1):
        """
        Initialize MetaARIMA with successive halving.

        Parameters:
        -----------
        config_space : List[str]
            List of ARIMA model configurations
        season_length : int
            Length of the seasonal cycle
        freq : str
            Time series frequency
        eta : float, default=2
            Controls the proportion of configurations to keep after each round
        resource_factor : float, default=2
            Controls how much the sample size increases in each round
        min_configs : int, default=1
            Minimum number of configurations to evaluate fully
        """
        super().__init__(config_space, season_length, freq)
        self.eta = eta
        self.resource_factor = resource_factor
        self.min_configs = min_configs
        self.init_resource_factor = init_resource_factor

    def _evaluate_models(self, df: pd.DataFrame, model_indices: List[int], sample_size: int) -> List[Tuple[int, float]]:
        """
        Evaluate a subset of models on a subset of data.

        Parameters:
        -----------
        df : pd.DataFrame
            The full training dataset
        model_indices : List[int]
            Indices of models to evaluate
        sample_size : int
            Number of time points to use in evaluation

        Returns:
        --------
        List[Tuple[int, float]]
            List of tuples (model_index, aicc_score)
        """
        df_subset = df.tail(sample_size).copy()
        print(df_subset.shape[0])

        models_subset = [self.models[i] for i in model_indices]
        sf_subset = StatsForecast(models=models_subset, freq=self.freq)

        sf_subset.fit(df=df_subset)

        # Get AICc scores
        results = []
        for i, model_idx in enumerate(model_indices):
            aicc = sf_subset.fitted_[0][i].model_['aicc']
            results.append((model_idx, aicc))

        return results

    def fit(self, df: pd.DataFrame):
        """
        Fit using successive halving to efficiently select the best model.
        """
        n_rows = len(df)
        n_models = self.nmodels

        s_max = int(log(n_models, self.eta))

        remaining_indices = list(range(n_models))

        min_sample_size = self.season_length * self.init_resource_factor

        for s in range(s_max + 1):
            sample_size = min(n_rows, int(min_sample_size * (self.resource_factor ** s)))

            eval_results = self._evaluate_models(df, remaining_indices, int(sample_size))

            eval_results.sort(key=lambda x: x[1])

            n_keep = max(int(n_models / (self.eta ** (s + 1))), self.min_configs)

            remaining_indices = [idx for idx, _ in eval_results[:n_keep]]

            if len(remaining_indices) <= 1:
                break

        best_idx = remaining_indices[0]
        self.sf = StatsForecast(models=[self.models[best_idx]], freq=self.freq)
        self.sf.fit(df=df)
        self.sf.fitted_[0][0].alias = self.alias


class MetaARIMABaseMC:

    def __init__(self,
                 config_space: List[str],
                 n_trials: int,
                 trial_n_obs: Union[int, float],
                 season_length: int,
                 freq: str):

        self.config_space = config_space
        self.freq = freq
        self.season_length = season_length
        self.n_trials = n_trials
        self.trial_n_obs = trial_n_obs
        self.models = None
        self.sf = None

        self.initialize_sf(self.config_space)
        self.nmodels = len(self.models)

        # self.models = MetaARIMAUtils.get_models_sf(season_length=self.season_length, alias_list=self.config_space)
        # self.sf = StatsForecast(models=self.models, freq=self.freq)

        self.alias = 'MetaARIMA'

    def fit(self, df: pd.DataFrame):
        aicc_trials = np.zeros((self.n_trials, self.nmodels))

        if self.trial_n_obs < 1:
            n_samples = max(int(len(df) * self.trial_n_obs), 20)
        else:
            n_samples = self.trial_n_obs

        for trial in range(self.n_trials):
            # Get a contiguous sample of n observations from df
            start_idx = np.random.randint(0, len(df) - n_samples + 1)
            sample_df = df.iloc[start_idx:start_idx + n_samples].copy()

            self.sf.fit(df=sample_df)

            # Store AICc values for this trial
            for i in range(self.nmodels):
                aicc_trials[trial, i] = self.sf.fitted_[0][i].model_['aicc']

        # Calculate average AICc across trials for each model
        avg_aicc = np.mean(aicc_trials, axis=0)

        # Select the model with lowest average AICc
        best_idx = avg_aicc.argmin()

        best_mod = self.sf.fitted_[0][best_idx]
        best_config = MetaARIMAUtils.get_model_order(best_mod.model_,
                                                     as_alias=True,
                                                     alias_freq=self.season_length)

        # Refit the best model with all data
        self.initialize_sf(config_space=[best_config])
        self.sf.fit(df=df)

        # self.sf.fitted_ = np.array([[self.sf.fitted_[0][best_idx]]])
        self.sf.fitted_[0][0].alias = self.alias

    def initialize_sf(self, config_space):
        self.models = MetaARIMAUtils.get_models_sf(season_length=self.season_length, alias_list=config_space)
        self.sf = StatsForecast(models=self.models, freq=self.freq)

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
                 use_mc: bool = False,
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
        self.use_mc = use_mc

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

        if self.use_mc:

            self.model = MetaARIMABaseMC(config_space=config_space,
                                         freq=self.freq,
                                         season_length=self.season_length, n_trials=10,
                                         trial_n_obs=0.4)
        else:
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
