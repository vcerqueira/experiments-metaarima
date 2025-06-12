from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform


from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.load_data.config import DATASETS

data_name, group = 'M3', 'Monthly'
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, extended=True)

train, _ = data_loader.train_test_split(df, horizon=horizon)

mdr = MetadataReader(dataset_name=data_name, group=group, freq_int=freq_int)

X, y, _, _, _ = mdr.read(fill_na_value=-1)
y = y.apply(lambda x: (x <= x.quantile(.1)).astype(int), axis=1)

# Define parameter distributions for random search
param_distributions = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 10),
    'num_leaves': randint(20, 150),
    'min_child_samples': randint(10, 100),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    # 'boosting_type': ['gbdt', 'dart', 'goss']
}

# Initialize base model
base_model = LGBMClassifier(verbosity=-1)

# Initialize RandomizedSearchCV with single validation split
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=50,  # Number of parameter settings sampled
    cv=3,  
    scoring='neg_log_loss',
    # scoring='roc_auc_ovr',
    # scoring='f1_weighted',
    # n_jobs=-1,
    random_state=42,
    verbose=2
)

# Fit RandomizedSearchCV
random_search.fit(X, y.iloc[:,0])


# Print best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best validation score:", random_search.best_score_)
