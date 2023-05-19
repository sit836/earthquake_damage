import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV

from constants import IN_PATH, OUT_PATH
from preprocessor import Preprocessor
from utils import plot_fe_importance


def make_submission(X_raw, y, is_tree, max_depth):
    X_test_raw = pd.read_csv(os.path.join(IN_PATH, 'test_values.csv'), index_col='building_id')

    ppc = Preprocessor()
    X = ppc.process(is_train=True, is_tree=is_tree, X=X_raw, y=y)
    X_test = ppc.process(is_train=False, is_tree=is_tree, X=X_test_raw, y=None)

    model = RandomForestClassifier(random_state=0, max_depth=max_depth)
    model.fit(X, y)
    pred_test = model.predict(X_test)

    submission_format = pd.read_csv(os.path.join(IN_PATH, 'submission_format.csv'), index_col='building_id')
    my_submission = pd.DataFrame(data=pred_test,
                                 columns=submission_format.columns,
                                 index=submission_format.index)
    my_submission.to_csv(os.path.join(OUT_PATH, 'submission.csv'))


def search_opt_model(X, y, model, param_grid):
    regressor = GridSearchCV(model, param_grid, cv=3, n_jobs=8)
    regressor.fit(X, y)
    print(regressor.best_params_)
    return regressor.best_estimator_


local_test = True

X_raw = pd.read_csv(os.path.join(IN_PATH, 'train_values.csv'), index_col='building_id')
train_labels = pd.read_csv(os.path.join(IN_PATH, 'train_labels.csv'), index_col='building_id')
y = train_labels.values.ravel()
is_tree = True

if local_test:
    X_train_raw, X_eval_raw, y_train, y_eval = train_test_split(X_raw, y, test_size=0.25, stratify=y, random_state=123)

    ppc = Preprocessor()
    X_train = ppc.process(is_train=True, is_tree=is_tree, X=X_train_raw, y=y_train)
    X_eval = ppc.process(is_train=False, is_tree=is_tree, X=X_eval_raw, y=None)
    print(f'X_train.shape: {X_train.shape}')

    param_grid = {"max_depth": [32, 64],
                  }
    model = RandomForestClassifier(random_state=0, max_depth=32, n_jobs=-1)
    # opt_gbm = search_opt_model(X_train, y_train, model,
    #                            param_grid=param_grid)
    # quit()

    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_eval = model.predict(X_eval)

    f1_train = f1_score(y_train, pred_train, average='micro')
    f1_eval = f1_score(y_eval, pred_eval, average='micro')
    print(f'f1_train, f1_eval: {(round(f1_train, 4), round(f1_eval, 4))}')

    plot_fe_importance(model, X_train.columns, num_top_fe=30)
else:
    make_submission(X_raw, y, is_tree, max_depth=32)
