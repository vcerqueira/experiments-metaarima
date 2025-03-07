from pprint import pprint
from pathlib import Path

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate

from src.load_data.config import DATASETS, DATA_GROUPS, GROUP_IDX
from src.arima.meta import MetaARIMAUtils

data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)

train, test = data_loader.train_test_split(df, horizon=horizon)

ORDER_MAX = {'AR': 4, 'I': 1, 'MA': 4, 'S_AR': 1, 'S_I': 1, 'S_MA': 1}
models = MetaARIMAUtils.get_models_sf(season_length=freq_int, max_config=ORDER_MAX)
# print(len(models))

if __name__ == '__main__':
    outfile = Path(__file__).parent.parent.parent.parent / 'assets' / 'metadata_cv'

    results = {}
    df_grouped = train.groupby('unique_id')
    for uid, uid_df in df_grouped:
        print(data_name, group, uid)
        if uid in results:
            continue

        # df = ds.query('unique_id=="Y1"')
        sf_auto = StatsForecast(models=[AutoARIMA(),
                                        SeasonalNaive(season_length=freq_int)],
                                freq=freq_str)
        sf_auto.fit(df=uid_df)
        fcst_auto = sf_auto.predict(h=horizon)

        arima_config = MetaARIMAUtils.get_model_order(sf_auto.fitted_[0][0].model_, as_alias=True, alias_freq=freq_int)

        sf = StatsForecast(models=models, freq=freq_str)
        sf.fit(df=uid_df)

        fcst = sf.predict(h=horizon)
        fcst = fcst.merge(test, on=['unique_id', 'ds'], how='left')
        fcst_auto = fcst_auto.merge(test, on=['unique_id', 'ds'], how='left')
        fcst = fcst.fillna(-1)

        err = evaluate(df=fcst, metrics=[smape]).mean(numeric_only=True)
        err_auto = evaluate(df=fcst_auto, metrics=[smape]).mean(numeric_only=True)
        err_auto_ = {'score_AutoARIMA': err_auto['AutoARIMA'],
                     'score_SeasNaive': err_auto['SeasonalNaive']}

        best_model_name = err.sort_values().index[0]

        best_model = sf.fitted_.flatten()[err.argmin()]

        # assert best_model.__str__() == best_model_name

        mod_summary = MetaARIMAUtils.model_summary(best_model.model_)

        pprint(mod_summary)

        uid_results = {**err.to_dict(),
                       **mod_summary,
                       **err_auto_,
                       'best_config': best_model_name,
                       'auto_config': arima_config,
                       'unique_id': uid}

        results[uid] = uid_results

        results_df = pd.DataFrame.from_dict(results).T
        type_dict = {col: float for col in results_df.columns if col not in ['best_config', 'auto_config', 'unique_id']}
        results_df = results_df.astype(type_dict)

        results_df.to_csv(f'{outfile}/arima,{data_name},{group}.csv', index=False)

    results_df = pd.DataFrame.from_dict(results).T
    type_dict = {col: float for col in results_df.columns if col not in ['best_config', 'auto_config', 'unique_id']}
    results_df = results_df.astype(type_dict)

    results_df.to_csv(f'{outfile}/arima,{data_name},{group}.csv', index=False)
