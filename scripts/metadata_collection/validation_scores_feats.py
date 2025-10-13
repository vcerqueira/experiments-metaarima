from pprint import pprint
from pathlib import Path

from tsfeatures import tsfeatures
import pandas as pd
from statsforecast import StatsForecast
from utilsforecast.losses import smape
from utilsforecast.evaluation import evaluate

from src.meta.arima._base import MetaARIMAUtils
from src.config import ORDER_MAX
from src.load_data.config import DATASETS

# data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
data_name, group = 'M4', 'Quarterly'
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=3)

train_all, _ = data_loader.train_test_split(df, horizon=horizon)
dev, validation = data_loader.train_test_split(train_all, horizon=horizon)

models = MetaARIMAUtils.get_models_sf(season_length=freq_int, max_config=ORDER_MAX)
print(len(models))

if __name__ == '__main__':
    outfile_features = Path(__file__).parent.parent.parent / 'assets' / 'features'
    outfile_arima = Path(__file__).parent.parent.parent / 'assets' / 'metadata_cv'

    feat_df = tsfeatures(dev, freq=freq_int, threads=6)

    feat_df.to_csv(f'{outfile_features}/features-dev,{data_name},{group}.csv', index=False)
    print(feat_df)

    results = {}
    df_grouped = dev.groupby('unique_id')
    for uid, uid_df in df_grouped:

        print(data_name, group, uid)

        sf = StatsForecast(models=models, freq=freq_str)
        try:
            sf.fit(df=uid_df)
        except ValueError:
            continue

        fcst = sf.predict(h=horizon)
        fcst = fcst.merge(validation, on=['unique_id', 'ds'], how='left')
        fcst = fcst.fillna(-1)

        err = evaluate(df=fcst, metrics=[smape]).mean(numeric_only=True)

        best_model_name = err.sort_values().index[0]

        best_model = sf.fitted_.flatten()[err.argmin()]

        mod_summary = MetaARIMAUtils.model_summary(best_model.model_)

        pprint(mod_summary)

        uid_results = {**err.to_dict(),
                       **mod_summary,
                       'best_config': best_model_name,
                       'dataset': f'{data_name},{group}',
                       'unique_id': uid}

        results[uid] = uid_results

        results_df = pd.DataFrame.from_dict(results).T
        type_dict = {col: float for col in results_df.columns if
                     col not in ['best_config', 'dataset', 'auto_config', 'unique_id']}
        results_df = results_df.astype(type_dict)

        results_df.to_csv(f'{outfile_arima}/arima-dev,{data_name},{group}.csv', index=False)

    results_df = pd.DataFrame.from_dict(results).T
    type_dict = {col: float for col in results_df.columns if
                 col not in ['best_config', 'dataset', 'auto_config', 'unique_id']}
    results_df = results_df.astype(type_dict)

    results_df.to_csv(f'{outfile_arima}/arima-dev,{data_name},{group}.csv', index=False)
