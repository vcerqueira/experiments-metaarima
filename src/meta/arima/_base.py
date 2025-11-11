import copy
import warnings
from typing import Optional, List, Dict, Union, Tuple
from math import log

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsforecast.models import ARIMA, MSTL
from statsforecast import StatsForecast
from tsfeatures import (acf_features, arch_stat, crossing_points,
                        entropy, flat_spots, heterogeneity,
                        holt_parameters, lumpiness, nonlinearity,
                        pacf_features, stl_features, stability,
                        hw_parameters, unitroot_kpss, unitroot_pp,
                        series_length, hurst, scalets)

from src.config import ORDER_MAX

warnings.filterwarnings(action='ignore')

FEATURE_ORDER = ['hurst', 'series_length', 'unitroot_pp', 'unitroot_kpss', 'hw_alpha', 'hw_beta', 'hw_gamma',
                 'stability', 'nperiods', 'seasonal_period', 'trend', 'spike', 'linearity', 'curvature', 'e_acf1',
                 'e_acf10', 'seasonal_strength', 'peak', 'trough', 'x_pacf5', 'diff1x_pacf5', 'diff2x_pacf5',
                 'seas_pacf', 'nonlinearity', 'lumpiness', 'alpha', 'beta', 'arch_acf', 'garch_acf', 'arch_r2',
                 'garch_r2', 'flat_spots', 'entropy', 'crossing_points', 'arch_lm', 'x_acf1', 'x_acf10', 'diff1_acf1',
                 'diff1_acf10', 'diff2_acf1', 'diff2_acf10', 'seas_acf1']

SEAS_FEATURES = ['seasonal_strength', 'peak', 'trough', 'seas_pacf', 'seas_acf1']
FEATURE_ORDER_NON_SEAS = [x for x in FEATURE_ORDER if x not in SEAS_FEATURES]


class _MetaARIMABase:

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

    def predict(self, h: int, level: Optional[List]):
        if level is not None:
            return self.sf.predict(h=h, level=level)
        else:
            return self.sf.predict(h=h)


class _HalvingMetaARIMABase(_MetaARIMABase):
    def __init__(self,
                 config_space: List[str],
                 season_length: int,
                 freq: str,
                 eval_mstl: bool = False,
                 eta: float = 2,
                 resource_factor: float = 2,
                 init_resource_factor: int = 4,
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
        self.tot_nobs = 0
        self.eval_mstl = eval_mstl

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
        # print(df_subset.shape[0])

        models_subset = [self.models[i] for i in model_indices]
        sf_subset = StatsForecast(models=models_subset, freq=self.freq)

        sf_subset.fit(df=df_subset)

        results = []
        for i, model_idx in enumerate(model_indices):
            aicc = sf_subset.fitted_[0][i].model_['aicc']
            results.append((model_idx, aicc))

        return results

    def fit(self, df: pd.DataFrame):
        """
        Fit using successive halving to efficiently select the best model.
        """
        assert self.nmodels > 0, 'nmodels is not positive'

        n_rows = df.shape[0]
        n_models = self.nmodels

        s_max = int(log(n_models, self.eta))

        remaining_indices = list(range(n_models))

        # todo hardcoded 20
        min_sample_size = self.season_length * self.init_resource_factor if self.season_length > 1 else 20

        self.tot_nobs = 0
        for s in range(s_max + 1):
            sample_size = min(n_rows, int(min_sample_size * (self.resource_factor ** s)))

            eval_results = self._evaluate_models(df, remaining_indices, int(sample_size))
            self.tot_nobs += len(remaining_indices) * int(sample_size)

            eval_results.sort(key=lambda x: x[1])

            n_keep = max(int(n_models / (self.eta ** (s + 1))), self.min_configs)

            remaining_indices = [idx for idx, _ in eval_results[:n_keep]]

            if len(remaining_indices) <= 1:
                break

        best_idx = remaining_indices[0]

        if self.eval_mstl:
            use_mstl = MSTLTestUtils.test_mstl_on_config(df=df,
                                                         config_inst=self.models[best_idx],
                                                         freq=self.freq)

            if use_mstl:
                trend_arima = ARIMA(order=self.models[best_idx].order,
                                    season_length=1,
                                    seasonal_order=(0, 0, 0))

                models = [MSTL(season_length=self.season_length, trend_forecaster=trend_arima)]

                self.sf = StatsForecast(models=models, freq=self.freq)
            else:
                self.sf = StatsForecast(models=[self.models[best_idx]], freq=self.freq)

            self.sf.fit(df=df)
            self.sf.fitted_[0][0].alias = self.alias
            self.tot_nobs += df.shape[0] * 3
        else:
            self.sf = StatsForecast(models=[self.models[best_idx]], freq=self.freq)
            self.sf.fit(df=df)
            self.tot_nobs += df.shape[0]
            self.sf.fitted_[0][0].alias = self.alias


class _MetaARIMABaseMC:

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

            try:
                self.sf.fit(df=sample_df)

                for i in range(self.nmodels):
                    aicc_trials[trial, i] = self.sf.fitted_[0][i].model_['aicc']
            except ValueError:
                for i in range(self.nmodels):
                    aicc_trials[trial, i] = 1e10

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
    # ORDER_MAX = {'AR': 2, 'I': 1, 'MA': 2, 'S_AR': 1, 'S_I': 1, 'S_MA': 1, }
    # ORDER_MAX = {'AR': 4, 'I': 1, 'MA': 4, 'S_AR': 1, 'S_I': 1, 'S_MA': 1}

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
            max_config_ = ORDER_MAX
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
                                          alias=f'ARIMA({ar},{i},{ma})({s_ar},{s_i},{s_ma})[{season_length}]',
                                          # prediction_intervals=ConformalIntervals(h=12)
                                          )
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


def tsfeatures_uid(uid_df: pd.DataFrame,
                   freq: int,
                   impute_seas: bool=False,
                   target_col: str = 'y',
                   id_col: str = 'unique_id'):
    x_r = uid_df[target_col].values
    x = scalets(x_r)

    feats = {
        **hurst(x, freq),
        **series_length(x, freq),
        **unitroot_pp(x, freq),
        **unitroot_kpss(x, freq),
        **hw_parameters(x, freq),
        **stability(x, freq),
        **stl_features(x, freq),
        **pacf_features(x, freq),
        **nonlinearity(x, freq),
        **lumpiness(x, freq),
        **holt_parameters(x, freq),
        **heterogeneity(x, freq),
        **flat_spots(x, freq),
        **entropy(x, freq),
        **crossing_points(x, freq),
        **arch_stat(x, freq),
        **acf_features(x, freq),
    }

    feats_series = pd.Series(feats)

    if impute_seas:
        for f in FEATURE_ORDER:
            if f not in feats_series.index:
                feats_series[f] = np.NaN

        feats_series = feats_series[FEATURE_ORDER]
    else:
        if freq > 1:
            feats_series = feats_series[FEATURE_ORDER]
        else:
            feats_series = feats_series[FEATURE_ORDER_NON_SEAS]

    feats_df = pd.DataFrame(feats_series).T.fillna(-1)
    feats_df.index = [uid_df[id_col].values[0]]

    return feats_df


class MSTLTestUtils:
    SEASONALITIES_BY_FREQ = {
        's': [60, 3600],
        'min': [60, 1440],
        '30T': [48, 336],
        'H': [24, 168],
        'D': [7, 365],
        'M': [4, 12],
        'ME': [4, 12],
        'MS': [4, 12],
    }

    @classmethod
    def test_mstl_on_config(cls, df: pd.DataFrame, config_inst, freq: str, max_samples: Optional[int] = None):
        assert freq in cls.SEASONALITIES_BY_FREQ, 'Unknown frequency. Check class'

        seas_l = cls.SEASONALITIES_BY_FREQ[freq]
        config_inst_ = copy.deepcopy(config_inst)

        df_ = df.copy()
        if max_samples is not None:
            df_ = df.tail(max_samples)

        trend_arima = ARIMA(order=config_inst_.order,
                            season_length=1,
                            seasonal_order=(0, 0, 0))

        models = [MSTL(season_length=seas_l, trend_forecaster=trend_arima),
                  config_inst_]

        sf_proxy = StatsForecast(models=models, freq=freq)

        sf_proxy.fit(df_)

        k_arima = cls.k_from_sf_arma(sf_proxy.fitted_[0][1].model_['arma'])

        sf_proxy.forecast(h=1, fitted=True)
        fitted_vals = sf_proxy.forecast_fitted_values()

        column_mapping = {col: 'ARIMA' for col in fitted_vals.columns
                          if col.startswith('ARIMA')
                          }

        fitted_vals = fitted_vals.rename(columns=column_mapping)

        resid_arima = fitted_vals['y'] - fitted_vals['ARIMA']
        resid_mstl = fitted_vals['y'] - fitted_vals['MSTL']

        aicc_arima = cls.compute_aicc_from_resids(resid_arima, k_arima)
        aicc_mstl = cls.compute_aicc_from_resids(resid_mstl, k_arima + 2)

        use_mstl = aicc_mstl < aicc_arima

        return use_mstl

    @staticmethod
    def compute_aicc_from_resids(resid: pd.Series, k):
        n = len(resid)
        # resid = fitted_vals['y']-fitted_vals['MSTL']
        ssr = np.sum(resid ** 2)
        sigma_hat_sq = ssr / n

        log_l_approx_term = n * np.log(sigma_hat_sq)

        penalty_term = 2 * k * (k + 1) / (n - k - 1)
        aicc = log_l_approx_term + 2 * k + penalty_term

        return aicc

    @staticmethod
    def k_from_sf_arma(arma):
        # k_arima = np.sum(sf.fitted_[0][1].model_['arma'][:4]) + 1
        return np.sum(arma[:4]) + 1
