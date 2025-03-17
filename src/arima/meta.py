from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.multioutput import ClassifierChain
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsforecast.models import ARIMA
from statsforecast import StatsForecast


class MetaARIMABase:

    def __init__(self, configs: List[str], season_length: int, freq: str):
        self.configs = configs
        self.freq = freq
        self.season_length = season_length
        self.models = MetaARIMAUtils.get_models_sf(season_length=self.season_length, alias_list=self.configs)
        self.n_models = len(self.models)
        self.sf = StatsForecast(models=self.models, freq=self.freq)

        self.alias = 'MetaARIMA'

    def fit(self, df: pd.DataFrame):
        self.sf.fit(df=df)

        aicc_ = [self.sf.fitted_[0][i].model_['aicc'] for i in range(self.n_models)]

        best_idx = np.array(aicc_).argmin()

        self.sf.fitted_ = np.array([[self.sf.fitted_[0][best_idx]]])
        self.sf.fitted_[0][0].alias = self.alias

    def predict(self, h: int):
        return self.sf.predict(h=h)


class MetaARIMAUtils:
    ORDER_MAX = {
        'AR': 2,
        'I': 1,
        'MA': 2,
        'S_AR': 1,
        'S_I': 1,
        'S_MA': 1,
    }

    @staticmethod
    def get_model_order(mod, as_alias: bool = False, alias_freq=1):
        order = tuple(mod["arma"][i] for i in [0, 5, 1, 2, 6, 3, 4])

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

        coefs = {f'coef_{k}': model['coef'][k]
                 for k in model['coef']}

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

        resid_tests = cls.test_residuals(model['residuals'])

        mod_summary = {**coefs, **goodness_fit, **resid_tests}

        return mod_summary

    @staticmethod
    def test_residuals(residuals):
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
        pvals = {}

        # Mean test
        mean_test = stats.ttest_1samp(residuals, 0)
        pvals['zero_mean'] = mean_test.pvalue

        # Homoscedasticity test
        squared_resid = residuals ** 2
        trend = np.arange(len(residuals))
        bp_test = stats.linregress(trend, squared_resid)
        pvals['constant_variance'] = bp_test.pvalue

        # Normality test
        jb_test = jarque_bera(residuals)
        pvals['normality'] = jb_test[1]

        # Autocorrelation test (for lag 1)
        lb_test = acorr_ljungbox(residuals, lags=[1])
        pvals['no_autocorrelation'] = lb_test.lb_pvalue.values[0]

        return pvals


class MetaARIMA:
    # todo compare against other arima implementations
    # todo eval meta-level
    # add mmr

    def __init__(self,
                 model: ClassifierChain,
                 freq: str,
                 season_length: int,
                 n_trials: int,
                 quantile_thr: float = 0.05,
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

        self.is_fit: bool = False

    def meta_fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        """

        :param X: feature set
        :param Y: error scores
        :return:
        """
        y = Y.apply(lambda x: (x <= x.quantile(self.quantile_thr)).astype(int), axis=1)
        self.model_names = Y.columns.tolist()
        if self.use_mmr:
            # self.corr_mat = compute_correlation_matrix(Y)
            self.corr_mat = Y.corr()

        self.meta_model.fit(X, y)
        self.is_fit = True

    def meta_predict(self, X):
        assert self.is_fit

        if self.use_mmr:

            meta_preds = self.meta_model.predict_proba(X)
            meta_preds = [pd.Series(x, index=self.model_names) for x in meta_preds]

            meta_preds_list = []
            for i, meta_pred in enumerate(meta_preds):
                selected_indices = mmr_selection(
                    probabilities=meta_pred,
                    correlation_matrix=self.corr_mat,
                    lambda_param=.5,
                    top_k=self.n_trials
                )

                mod_list = meta_pred.index[selected_indices].tolist()

                meta_preds_list.append(mod_list)
        else:
            preds = pd.DataFrame(self.meta_model.predict_proba(X), columns=self.model_names)

            meta_preds_list = preds.apply(lambda x: x.sort_values().index[:self.n_trials].tolist(),
                                          axis=1).values.tolist()

        return meta_preds_list

    def fit(self, df: pd.DataFrame, config_list: List[str]):
        assert self.is_fit

        self.model = MetaARIMABase(configs=config_list,
                                   freq=self.freq,
                                   season_length=self.season_length)
        self.model.fit(df)

    @classmethod
    def from_model(cls, n_trials: int):
        # instance from saved model
        pass


def mmr_selection(probabilities, correlation_matrix, lambda_param=0.5, top_k=10):
    """
    Select configurations using Maximal Marginal Relevance to balance between
    probability of being in the top 10% and diversity (low correlation with already selected configurations).

    Parameters:
    -----------
    probabilities : array-like, shape (n_configurations,)
        The predicted probabilities of each configuration being in the top 10%.
    correlation_matrix : array-like, shape (n_configurations, n_configurations)
        The correlation matrix between configurations.
    lambda_param : float, default=0.5
        The trade-off parameter. Higher values favor probability, lower values favor diversity.
        Range: [0, 1]
    top_k : int, default=10
        The number of configurations to select.

    Returns:
    --------
    selected_indices : list
        Indices of the selected configurations.
    """
    n_configs = len(probabilities)

    # Ensure lambda is within valid range
    lambda_param = max(0, min(1, lambda_param))

    # Initialize the list of selected configurations
    selected_indices = []

    # Set of remaining candidates
    remaining_indices = set(range(n_configs))

    # Select the first configuration (highest probability)
    best_idx = np.argmax(probabilities)
    selected_indices.append(best_idx)
    remaining_indices.remove(best_idx)

    # Select the rest using MMR
    while len(selected_indices) < top_k and remaining_indices:
        max_mmr_score = -float('inf')
        next_best_idx = None

        for idx in remaining_indices:
            # Compute the maximum correlation with already selected configurations
            max_correlation = max([correlation_matrix.iloc[idx, sel_idx] for sel_idx in selected_indices])

            # Compute MMR score
            # High lambda: favor high probability
            # Low lambda: favor low correlation (diversity)
            mmr_score = lambda_param * probabilities.iloc[idx] - (1 - lambda_param) * max_correlation

            if mmr_score > max_mmr_score:
                max_mmr_score = mmr_score
                next_best_idx = idx

        if next_best_idx is not None:
            selected_indices.append(next_best_idx)
            remaining_indices.remove(next_best_idx)
        else:
            break

    return selected_indices
