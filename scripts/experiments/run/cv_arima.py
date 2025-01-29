import os
from pprint import pprint

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from scipy import stats
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate
from tsfeatures import tsfeatures

from utils.load_data.config import DATASETS, DATA_GROUPS, GROUP_IDX

data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df = data_loader.get_uid_tails(df, tail_size=100)

train, test = data_loader.train_test_split(df, horizon=horizon)


class MetaARIMAUtils:
    ORDER_MAX = {
        'AR': 2,
        'I': 1,
        'MA': 2,
        'S_AR': 1,
        'S_I': 1,
        'S_MA': 1,
    }

    @classmethod
    def get_models_sf(cls):
        models = []
        for ar in range(cls.ORDER_MAX['AR'] + 1):
            for i in range(cls.ORDER_MAX['I'] + 1):
                for ma in range(cls.ORDER_MAX['MA'] + 1):
                    for s_ar in range(cls.ORDER_MAX['S_AR'] + 1):
                        for s_i in range(cls.ORDER_MAX['S_I'] + 1):
                            for s_ma in range(cls.ORDER_MAX['S_MA'] + 1):
                                models.append(
                                    ARIMA(order=(ar, i, ma),
                                          season_length=freq_int,
                                          seasonal_order=(s_ar, s_i, s_ma),
                                          alias=f'ARIMA({ar},{i},{ma})({s_ar},{s_i},{s_ma})[{freq_int}]')
                                )

        return models

    @staticmethod
    def get_model_order(mod):
        # obsolete
        order = tuple(mod["arma"][i] for i in [0, 5, 1, 2, 6, 3, 4])

        order_s = pd.Series(order, index=['AR', 'I', 'MA', 'S_AR', 'S_I', 'S_MA', 'm'])

        return order_s

    @classmethod
    def model_summary(cls, model):

        coefs = {f'coef_{k}': model['coef'][k]
                 for k in model['coef']}

        gdnss_fit = {
            'var_coef_mean': model['var_coef'].mean(),
            'aic': model['aic'],
            'aicc': model['aicc'],
            'bic': model['bic'],
            'loglik': model['loglik'],
        }

        resid_tests = cls.test_residuals(model['residuals'])

        mod_summary = {**coefs, **gdnss_fit, **resid_tests}

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


models = MetaARIMAUtils.get_models_sf()

results = {}

df_grouped = train.groupby('unique_id')
for uid, uid_df in df_grouped:
    print(data_name, group, uid)
    # df = ds.query('unique_id=="Y1"')

    sf = StatsForecast(models=models, freq=freq_str)

    sf.fit(df=uid_df)

    fcst = sf.predict(h=horizon)
    fcst = fcst.merge(test, on=['unique_id', 'ds'], how='left')

    err = evaluate(df=fcst, metrics=[smape]).mean(numeric_only=True)

    best_model_name = err.sort_values().index[0]

    best_model = sf.fitted_.flatten()[err.argmin()]

    assert best_model.__str__() == best_model_name

    mod_summary = MetaARIMAUtils.model_summary(best_model.model_)

    pprint(mod_summary)

    uid_results = {**err.to_dict(), **mod_summary, 'best_config': best_model_name, 'unique_id': uid}

    results[uid] = uid_results
