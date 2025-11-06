from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader, ModelIO
from src.chronos_data import ChronosDataset
from src.config import (MMR,
                        N_TRIALS,
                        QUANTILE_THR,
                        BASE_OPTIM,
                        LAMBDA,
                        PCA_N_COMPONENTS)

FILENAME = 'assets/trained_metaarima_m4m.joblib.gz'
# -- train metamodel
source = 'm4_monthly'
_, _, _, freq_str, freq_int = ChronosDataset.load_everything(source)

mdr = MetadataReader(group=source, freq_int=freq_int)
X, y, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1)

# model = CatBoostRegressor(
#     loss_function="MultiRMSE",
#     eval_metric="MultiRMSE",
#     iterations=300,
#     depth=3,
#     learning_rate=0.05,
#     random_seed=42,
# )

BEST_LGBM_PARAMS = {'colsample_bytree': 0.8468358280832688,
                    'learning_rate': 0.10856504541106007,
                    'max_depth': 4,
                    'min_child_samples': 77,
                    'n_estimators': 282,
                    'num_leaves': 142,
                    'subsample': 0.7726479814832277}

model = MultiOutputRegressor(LGBMRegressor(**BEST_LGBM_PARAMS))

# model = XGBRFRegressor()

meta_arima = MetaARIMA(model=model,
                       freq=freq_str,
                       season_length=freq_int,
                       n_trials=N_TRIALS,
                       quantile_thr=QUANTILE_THR,
                       pca_n_components=PCA_N_COMPONENTS,
                       use_mmr=MMR,
                       base_optim=BASE_OPTIM,
                       mmr_lambda=LAMBDA)

meta_arima.meta_fit(X, y)

ModelIO.save_model(meta_arima, FILENAME)

##

import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# BEST_CATBOOST_PARAMS = {'depth': 5,
#                         'learning_rate': 0.11292956428752265,
#                         'l2_leaf_reg': 17.43840587120914,
#                         'model_size_reg': 2.4988982836088725,
#                         'border_count': 64,
#                         'subsample': 0.6622965490826499,
#                         'rsm': 0.8392252548382597,
#                         'leaf_estimation_iterations': 4,
#                         'iterations': 889,
#                         'od_wait': 60}

BEST_CATBOOST_PARAMS = {'iterations': 889, 'learning_rate': 0.11292956428752265, 'depth': 5,
                        'l2_leaf_reg': 17.43840587120914, 'model_size_reg': 2.4988982836088725,
                        'rsm': 0.8392252548382597, 'loss_function': 'MultiRMSE', 'border_count': 64, 'od_wait': 60,
                        'od_type': 'Iter', 'leaf_estimation_iterations': 4, 'random_seed': 42,
                        'use_best_model': False,
                        'verbose': False, 'eval_metric': 'MultiRMSE', 'task_type': 'CPU', 'bootstrap_type': 'Bernoulli',
                        'subsample': 0.6622965490826499}


def tune_catboost_small(X, y, n_trials=30, random_state=42):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=random_state)
    train_pool, valid_pool = Pool(X_tr, y_tr), Pool(X_va, y_va)

    def objective(trial):
        params = {
            "loss_function": "MultiRMSE",
            "eval_metric": "MultiRMSE",
            "random_seed": random_state,
            "task_type": "CPU",
            "depth": trial.suggest_int("depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 60.0, log=True),
            "model_size_reg": trial.suggest_float("model_size_reg", 0.2, 2.5),
            "border_count": trial.suggest_int("border_count", 32, 64, step=32),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "rsm": trial.suggest_float("rsm", 0.6, 0.95),
            # "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 4),
            "bootstrap_type": "Bernoulli",
            "iterations": trial.suggest_int("iterations", 300, 900),
            "od_type": "Iter",
            "od_wait": trial.suggest_int("od_wait", 20, 80, step=10),
            "use_best_model": True,
            "verbose": False
        }
        m = CatBoostRegressor(**params)
        m.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)
        return m.get_best_score()["validation"]["MultiRMSE"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print('study.best_params')
    print(study.best_params)

    best = study.best_params
    final = CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        random_seed=random_state,
        task_type="CPU",
        depth=best["depth"],
        learning_rate=best["learning_rate"],
        l2_leaf_reg=best["l2_leaf_reg"],
        model_size_reg=best["model_size_reg"],
        border_count=best["border_count"],
        subsample=best["subsample"],
        rsm=best["rsm"],
        # bagging_temperature=best["bagging_temperature"],
        leaf_estimation_iterations=best["leaf_estimation_iterations"],
        bootstrap_type="Bernoulli",
        iterations=best["iterations"],
        od_type="Iter",
        od_wait=best["od_wait"],
        use_best_model=True,
        verbose=False
    )
    final.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
    if getattr(final, "best_iteration_", None):
        final.shrink(ntree_end=final.best_iteration_)

    return final, study.best_value, study.best_params


optimized_model.get_params()

optimized_model, valid_multi_rmse, optimized_params = tune_catboost_small(X, y, n_trials=30, random_state=42)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=random_state)

model = CatBoostRegressor(
    # loss_function="MultiRMSE",
    # eval_metric="MultiRMSE",
    # random_seed=42,
    **BEST_CATBOOST_PARAMS,
    # use_best_model=False,
    # verbose=False
)

meta_arima = MetaARIMA(
    model=model,
    freq=freq_str,
    season_length=freq_int,
    n_trials=N_TRIALS,
    quantile_thr=QUANTILE_THR,
    pca_n_components=PCA_N_COMPONENTS,
    use_mmr=MMR,
    base_optim=BASE_OPTIM,
    mmr_lambda=LAMBDA
)

meta_arima.meta_fit(X, y)
