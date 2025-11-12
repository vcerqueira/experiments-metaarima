import copy
from pprint import pprint

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.multioutput import ClassifierChain, RegressorChain, MultiOutputRegressor
from utilsforecast.losses import mase
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.chronos_data import ChronosDataset
from src.config import (N_TRIALS,
                        BASE_OPTIM,
                        LAMBDA,
                        PCA_N_COMPONENTS,
                        ORDER_MAX,
                        QUANTILE_THR,
                        BEST_CATBOOST_PARAMS)

# -- train metamodel
algorithm = 'catboost'
source = 'm4_monthly'
FILENAME = f'assets/trained_metaarima_{source}_{algorithm}.joblib.gz'
_, _, _, freq_str, freq_int = ChronosDataset.load_everything(source)

mdr = MetadataReader(group=source, freq_int=freq_int)
X, y, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1, max_config=ORDER_MAX)

cb_params_multi = copy.deepcopy(BEST_CATBOOST_PARAMS[source])
cb_params_uni = copy.deepcopy(BEST_CATBOOST_PARAMS[source])
cb_params_uni['loss_function'] = 'RMSE'
cb_params_uni['eval_metric'] = 'RMSE'

cb_params_clf = copy.deepcopy(BEST_CATBOOST_PARAMS[source])
cb_params_clf['loss_function'] = 'Logloss'
cb_params_clf['eval_metric'] = 'Logloss'

pprint(cb_params_multi)
pprint(cb_params_uni)
pprint(cb_params_clf)

mod = CatBoostRegressor(**cb_params_multi)

mod_clf_ch = ClassifierChain(CatBoostClassifier(**cb_params_clf))
mod_reg_ch = RegressorChain(CatBoostRegressor(**cb_params_uni))
mod_reg_mo = MultiOutputRegressor(CatBoostRegressor(**cb_params_uni))

meta_arima_d = {}

meta_arima_d['MetaARIMA'] = MetaARIMA(model=mod,
                                      freq=freq_str,
                                      season_length=freq_int,
                                      n_trials=N_TRIALS,
                                      pca_n_components=PCA_N_COMPONENTS,
                                      quantile_thr=QUANTILE_THR,
                                      use_mmr=True,
                                      base_optim=BASE_OPTIM,
                                      mmr_lambda=LAMBDA)

meta_arima_d['No-PCA'] = MetaARIMA(model=mod_clf_ch,
                                   freq=freq_str,
                                   season_length=freq_int,
                                   n_trials=N_TRIALS,
                                   target_pca=False,
                                   pca_n_components=PCA_N_COMPONENTS,
                                   quantile_thr=QUANTILE_THR,
                                   use_mmr=True,
                                   base_optim=BASE_OPTIM,
                                   mmr_lambda=LAMBDA)

meta_arima_d['Reg-Chain'] = MetaARIMA(model=mod_reg_ch,
                                      freq=freq_str,
                                      season_length=freq_int,
                                      n_trials=N_TRIALS,
                                      target_pca=True,
                                      pca_n_components=PCA_N_COMPONENTS,
                                      quantile_thr=QUANTILE_THR,
                                      use_mmr=True,
                                      base_optim=BASE_OPTIM,
                                      mmr_lambda=LAMBDA)

meta_arima_d['Meta-Regr'] = MetaARIMA(model=mod,
                                      freq=freq_str,
                                      season_length=freq_int,
                                      n_trials=N_TRIALS,
                                      meta_regression=True,
                                      pca_n_components=PCA_N_COMPONENTS,
                                      quantile_thr=QUANTILE_THR,
                                      use_mmr=True,
                                      base_optim=BASE_OPTIM,
                                      mmr_lambda=LAMBDA)

meta_arima_d['No-SH'] = MetaARIMA(model=mod,
                                  freq=freq_str,
                                  season_length=freq_int,
                                  n_trials=N_TRIALS,
                                  pca_n_components=PCA_N_COMPONENTS,
                                  quantile_thr=QUANTILE_THR,
                                  use_mmr=True,
                                  base_optim='complete',
                                  mmr_lambda=LAMBDA)

meta_arima_d['MonteCarlo'] = MetaARIMA(model=mod,
                                       freq=freq_str,
                                       season_length=freq_int,
                                       n_trials=N_TRIALS,
                                       quantile_thr=QUANTILE_THR,
                                       pca_n_components=PCA_N_COMPONENTS,
                                       use_mmr=True,
                                       base_optim='mc',
                                       mmr_lambda=LAMBDA)

meta_arima_d['MO-Regr'] = MetaARIMA(model=mod_reg_mo,
                                    freq=freq_str,
                                    season_length=freq_int,
                                    n_trials=N_TRIALS,
                                    quantile_thr=QUANTILE_THR,
                                    pca_n_components=PCA_N_COMPONENTS,
                                    use_mmr=True,
                                    base_optim=BASE_OPTIM,
                                    mmr_lambda=LAMBDA)

meta_arima_d['No-MMR'] = MetaARIMA(model=mod,
                                   freq=freq_str,
                                   season_length=freq_int,
                                   n_trials=N_TRIALS,
                                   pca_n_components=PCA_N_COMPONENTS,
                                   quantile_thr=QUANTILE_THR,
                                   use_mmr=False,
                                   base_optim=BASE_OPTIM,
                                   mmr_lambda=LAMBDA)

for variant_ in meta_arima_d:
    print(variant_)
    meta_arima_d[variant_].meta_fit(X, y)
    # meta_arima_d[variant_].meta_fit(X.head(400), y.head(400))

model_names = [*meta_arima_d] + ['AutoARIMA']

target = 'monash_m3_monthly'
# target = 'monash_hospital'
df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
train, test = ChronosDataset.time_wise_split(df, horizon)

sf_models = [AutoARIMA(season_length=seas_len)]

uids = train['unique_id'].unique().tolist()

results = []
for uid in uids:
    print(uid)

    df_uid_tr = train.query(f'unique_id=="{uid}"').reset_index(drop=True)
    df_uid_ts = test.query(f'unique_id=="{uid}"').reset_index(drop=True)

    meta_arima_fcst = {}
    for ma_k, ma_v in meta_arima_d.items():
        # print(ma_k)
        ma_v.fit(df_uid_tr, freq=freq, seas_length=seas_len)

        fcst_ma_ = ma_v.predict(h=horizon)

        meta_arima_fcst[ma_k] = fcst_ma_['MetaARIMA'].values

    fcst_meta_arima = pd.DataFrame(meta_arima_fcst)

    sf = StatsForecast(models=copy.deepcopy(sf_models), freq=freq)
    sf.fit(df_uid_tr)

    fcst_aa = sf.forecast(h=horizon).reset_index()
    fcst_meta_arima['ds'] = fcst_aa['ds']
    fcst_meta_arima['unique_id'] = fcst_aa['unique_id']

    uid_test = df_uid_ts.merge(fcst_meta_arima, on=['unique_id', 'ds'])
    uid_test = uid_test.merge(fcst_aa, on=['unique_id', 'ds'])

    err = mase(df=uid_test, models=model_names, seasonality=seas_len, train_df=df_uid_tr)

    pprint(err)

    results.append(err)
    results_df = pd.concat(results)
    print(results_df.mean(numeric_only=True))
    print(results_df.median(numeric_only=True))

results_df = pd.concat(results)
print(results_df.mean(numeric_only=True))
print(results_df.median(numeric_only=True))

results_df.to_csv(f'assets/results/main/ablation,{target}.csv', index=False)
