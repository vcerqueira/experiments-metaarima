from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.load_data.config import DATASETS
from src.config import MMR, N_TRIALS, QUANTILE_THR, BASE_OPTIM, LAMBDA

data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
# data_name, group = 'Tourism', 'Quarterly'
# data_name, group = 'M4', 'Monthly'
# data_name, group = 'M4', 'Weekly'
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, extended=True)

train, _ = data_loader.train_test_split(df, horizon=horizon)

mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=freq_int)

X, y, _, _, cv = mdr.read(fill_na_value=-1)

# note that this is cv on the time series set (80% of time series for train, 20% for testing)
# partition is done at time series level, not in time dimension
kfcv = KFold(n_splits=5, random_state=1, shuffle=True)

results = []
for j, (train_index, test_index) in enumerate(kfcv.split(X)):
    print(f"Fold {j}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

    X_train = X.iloc[train_index, :]
    y_train = y.iloc[train_index, :]
    X_test = X.iloc[test_index, :]
    y_test = y.iloc[test_index, :]

    mod = ClassifierChain(LGBMClassifier(verbosity=-1))

    meta_arima = MetaARIMA(model=mod,
                           freq=freq_str,
                           season_length=freq_int,
                           n_trials=N_TRIALS,
                           quantile_thr=QUANTILE_THR,
                           use_mmr=MMR,
                           base_optim=BASE_OPTIM,
                           mmr_lambda=LAMBDA)

    meta_arima.meta_fit(X_train, y_train)
    pred_list = meta_arima.meta_predict(X_test)

    for i, (uid, x) in enumerate(X_test.iterrows()):
        print(i, uid)

        df_uid = train.query(f'unique_id=="{uid}"').copy()

        meta_arima.fit(df_uid, config_space=pred_list[i])
        auto_arima_config = cv.loc[uid, 'auto_config']

        err_meta = cv.loc[uid, meta_arima.selected_config]
        err_auto = cv.loc[uid, 'score_AutoARIMA']
        err_snaive = cv.loc[uid, 'score_SeasNaive']
        err_theta = cv.loc[uid, 'score_AutoTheta']
        err_ets = cv.loc[uid, 'score_AutoETS']

        config212 = f'ARIMA(2,1,2)(0,0,0)[{freq_int}]'
        config2121 = f'ARIMA(2,1,2)(1,0,0)[{freq_int}]'
        config100 = f'ARIMA(1,0,0)(0,0,0)[{freq_int}]'

        err_arima212 = cv.loc[uid, config212]
        err_arima2121 = cv.loc[uid, config2121]
        err_arima100 = cv.loc[uid, config100]

        try:
            err_auto2 = cv.loc[uid, auto_arima_config]
        except KeyError:
            err_auto2 = np.nan

        comp = {
            'MetaARIMA': err_meta,
            'AutoARIMA': err_auto,
            'AutoARIMA2': err_auto2,
            'ARIMA(2,1,2)': err_arima212,
            'ARIMA(2,1,2)(1,0,0)': err_arima2121,
            'ARIMA(1,0,0)': err_arima100,
            'SeasonalNaive': err_snaive,
            'AutoTheta': err_theta,
            'AutoETS': err_ets,
        }

        pprint(comp)

        results.append(comp)

results_df = pd.DataFrame(results)

results_df.to_csv(f'assets/results/{data_name},{group}.csv',index=False)

# results_df = pd.read_csv(f'assets/results/{data_name},{group}.csv')

a=results_df.copy()
b=results_df.copy()

res = pd.concat([a,b]).reset_index(drop=True).drop(columns=['AutoARIMA2','ARIMA(2,1,2)(1,0,0)','AutoTheta','AutoETS'])
print(res.mean())
print(res.median())
print(res.rank(axis=1).mean())

import pandas as pd
from plotnine import ggplot, element_text,aes, geom_bar, theme,theme_minimal, labs, coord_flip

# Your data as a pandas Series
s = pd.Series({
    "MetaARIMA": 2.439560,
    "AutoARIMA": 2.744048,
    "ARIMA(2,1,2)": 2.786172,
    "ARIMA(1,0,0)": 3.609432,
    "SeasonalNaive": 3.420788
})

# Convert to DataFrame for plotnine
df = s.reset_index()
df.columns = ['Method', 'AverageRank']

# Create the barplot
p = (
    ggplot(df, aes(x='reorder(Method, AverageRank)', y='AverageRank'))
    + geom_bar(stat='identity', fill='#4C72B0')
    # + coord_flip()  # Optional: horizontal bars
    + theme_minimal()
    + theme(axis_text=element_text(size=12))
    + labs(
        x='',
        y='Average Rank',
        title=''
    )
)

print(p)
p.save('avg_rank.pdf', width=12,height=4.5)


print(results_df.mean())
print(results_df.median())
print(results_df.rank(axis=1).mean())
print(results_df.dropna().mean())
print(results_df.dropna().median())
print(results_df.dropna().rank(axis=1).mean())
