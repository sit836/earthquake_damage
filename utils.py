import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from constants import BINARY_COLS, MULTIARY_COLS, GEO_LVLS


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def plot_fe_importance(model, feature_names, num_top_fe):
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances.sort_values(ascending=False, inplace=True)
    forest_importances = forest_importances[:num_top_fe]

    fig, ax = plt.subplots()
    sns.barplot(x=forest_importances.values, y=forest_importances.index)
    fig.tight_layout()
    plt.show()


def tune_lgbm(X_train, y_train, X_eval, y_eval, n_trials=20):
    def objective(trial):
        param = {
            "objective": "gini",
            "metric": "multi_logloss",
            "num_class": 3,
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 32, 1024),
            "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_child_samples", 20, 10000),
        }

        model = lgb.LGBMClassifier(**param, random_state=0, n_jobs=-1)
        model.fit(X_train, y_train)
        pred_eval = model.predict(X_eval)
        return f1_score(y_eval, pred_eval, average='micro')

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(f'trial.params: {trial.params}')


def tune_rf(X_train, y_train, X_eval, y_eval, n_trials=20):
    def objective(trial):
        param = {
            "max_depth": trial.suggest_int("max_depth", 2, 256),
            "max_features": trial.suggest_float("max_features", 0.001, 0.5),
        }

        model = RandomForestClassifier(**param, random_state=0, n_jobs=-1)
        model.fit(X_train, y_train)
        pred_eval = model.predict(X_eval)
        return f1_score(y_eval, pred_eval, average='micro')

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(f'trial.params: {trial.params}')


def tune_catboost(X_train, y_train, X_eval, y_eval, n_trials=20):
    def objective(trial):
        param = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 10),
            "random_strength": trial.suggest_float("random_strength", 0., 10.),
        }

        model = CatBoostClassifier(**param, cat_features=BINARY_COLS + MULTIARY_COLS + GEO_LVLS, random_state=0)
        model.fit(X_train, y_train)
        pred_eval = model.predict(X_eval)
        return f1_score(y_eval, pred_eval, average='micro')

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(f'trial.params: {trial.params}')


def cpu():
    return torch.device('cpu')


def gpu(i=0):
    return torch.device(f'cuda:{i}')


def num_gpus():
    return torch.cuda.device_count()


def try_gpu(i=0):
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()
