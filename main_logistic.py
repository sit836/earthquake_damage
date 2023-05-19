import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from constants import IN_PATH, OUT_PATH
from preprocessor import Preprocessor


def make_submission(X_raw, y, is_tree):
    X_test_raw = pd.read_csv(os.path.join(IN_PATH, 'test_values.csv'), index_col='building_id')

    ppc = Preprocessor()
    X = ppc.process(is_train=True, is_tree=is_tree, X=X_raw, y=y)
    X_test = ppc.process(is_train=False, is_tree=is_tree, X=X_test_raw, y=None)

    model = LogisticRegression(random_state=0)
    model.fit(X, y)
    pred_test = model.predict(X_test)

    submission_format = pd.read_csv(os.path.join(IN_PATH, 'submission_format.csv'), index_col='building_id')
    my_submission = pd.DataFrame(data=pred_test,
                                 columns=submission_format.columns,
                                 index=submission_format.index)
    my_submission.to_csv(os.path.join(OUT_PATH, 'submission.csv'))


local_test = True

X_raw = pd.read_csv(os.path.join(IN_PATH, 'train_values.csv'), index_col='building_id')
train_labels = pd.read_csv(os.path.join(IN_PATH, 'train_labels.csv'), index_col='building_id')
y = train_labels.values.ravel()
is_tree = False

if local_test:
    X_train_raw, X_eval_raw, y_train, y_eval = train_test_split(X_raw, y, test_size=0.25, stratify=y, random_state=123)

    ppc = Preprocessor()
    X_train = ppc.process(is_train=True, is_tree=is_tree, X=X_train_raw, y=y_train)
    X_eval = ppc.process(is_train=False, is_tree=is_tree, X=X_eval_raw, y=None)
    print(f'X_train.shape: {X_train.shape}')

    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_eval = model.predict(X_eval)

    f1_train = f1_score(y_train, pred_train, average='micro')
    f1_eval = f1_score(y_eval, pred_eval, average='micro')
    print(f'f1_train, f1_eval: {(round(f1_train, 4), round(f1_eval, 4))}')
else:
    make_submission(X_raw, y, is_tree)
