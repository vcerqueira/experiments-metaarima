from pprint import pprint

import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.meta.arima.meta_arima import MetaARIMA
from src.meta.arima._data_reader import MetadataReader
from src.chronos_data import ChronosDataset
from src.config import QUANTILE_THR, PCA_N_COMPONENTS

source = 'm4_monthly'
_, _, _, freq_str, freq_int = ChronosDataset.load_everything(source)

mdr = MetadataReader(group=source, freq_int=freq_int)
X, y, _, _, _ = mdr.read(from_dev_set=True, fill_na_value=-1)

y = y.apply(lambda x: (x <= x.quantile(QUANTILE_THR)).astype(int), axis=1)
pca = PCA(n_components=PCA_N_COMPONENTS)
y_pca = pca.fit_transform(y)


# model = CatBoostRegressor(
#     loss_function="MultiRMSE",
#     eval_metric="MultiRMSE",
#     iterations=300,
#     depth=3,
#     learning_rate=0.05,
#     random_seed=42,
# )


def tune_catboost_small(X, y, n_trials=30, random_state=42):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=random_state)
    train_pool, valid_pool = Pool(X_tr, y_tr), Pool(X_va, y_va)

    def objective(trial):
        params = {
            "loss_function": "MultiRMSE",
            "eval_metric": "MultiRMSE",
            "random_seed": random_state,

            "task_type": "CPU",
            "depth": trial.suggest_int("depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 60.0, log=True),
            "model_size_reg": trial.suggest_float("model_size_reg", 0.2, 2.5),
            "border_count": trial.suggest_int("border_count", 32, 64, step=32),
            "rsm": trial.suggest_float("rsm", 0.6, 0.95),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 5),
            "bootstrap_type": "Bernoulli",
            "iterations": trial.suggest_int("iterations", 100, 1100, step=100),
            "od_type": "Iter",
            "od_wait": trial.suggest_int("od_wait", 20, 80, step=10),
            "use_best_model": False,
            "verbose": False
        }

        m = CatBoostRegressor(**params)
        m.fit(train_pool, eval_set=valid_pool)
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
        rsm=best["rsm"],
        leaf_estimation_iterations=best["leaf_estimation_iterations"],
        bootstrap_type="Bernoulli",
        iterations=best["iterations"],
        od_type="Iter",
        od_wait=best["od_wait"],
        use_best_model=False,
        verbose=False
    )

    final.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    if getattr(final, "best_iteration_", None):
        final.shrink(ntree_end=final.best_iteration_)

    return final, study.best_params


opt_model, best_params = tune_catboost_small(X, y_pca,
                                             n_trials=50,
                                             random_state=42)

pprint(opt_model.get_params())

BEST_CATBOOST_PARAMS = {'bootstrap_type': 'Bernoulli',
                        'border_count': 32,
                        'depth': 4,
                        'eval_metric': 'MultiRMSE',
                        'iterations': 283,
                        'l2_leaf_reg': 16.732679992180078,
                        'leaf_estimation_iterations': 2,
                        'learning_rate': 0.04248616344796388,
                        'loss_function': 'MultiRMSE',
                        'model_size_reg': 2.0916597401644705,
                        'od_type': 'Iter',
                        'od_wait': 60,
                        'random_seed': 42,
                        'rsm': 0.6406208143252348,
                        'task_type': 'CPU',
                        'use_best_model': False,
                        'verbose': False}

BEST_CATBOOST_PARAMS = {'bootstrap_type': 'Bernoulli',
                        'border_count': 64,
                        'depth': 4,
                        'eval_metric': 'MultiRMSE',
                        'iterations': 1000,
                        'l2_leaf_reg': 44.766640345162486,
                        'leaf_estimation_iterations': 3,
                        'learning_rate': 0.05542045515461307,
                        'loss_function': 'MultiRMSE',
                        'model_size_reg': 2.490479898592805,
                        'od_type': 'Iter',
                        'od_wait': 50,
                        'random_seed': 42,
                        'rsm': 0.8036450022637952,
                        'task_type': 'CPU',
                        'use_best_model': False,
                        'verbose': False}
