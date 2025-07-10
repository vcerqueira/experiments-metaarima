# todo create model based on m4

from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier
from xgboost import XGBRFClassifier

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.load_data.config import DATASETS
from src.config import MMR, N_TRIALS, QUANTILE_THR, BASE_OPTIM, LAMBDA

# data_name, group = 'M3', 'Monthly'
# data_name, group = 'M3', 'Quarterly'
# data_name, group = 'Tourism', 'Monthly'
data_name, group = 'Tourism', 'Quarterly'
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
mod = ClassifierChain(LGBMClassifier(verbosity=-1))
mod = XGBRFClassifier()
y_train_ = (y_train < 0.15).astype(int)
mod.fit(X_train, y_train_)

# ----

import joblib

filename = 'model_example.sav'
joblib.dump(mod, filename)

loaded_model = joblib.load(filename)

preds = loaded_model.predict_proba(X_test)
# preds = mod.predict(X_test)
print(preds)
