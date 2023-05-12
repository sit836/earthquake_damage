import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from constants import IN_PATH
from preprocessor import Preprocessor
from submit import make_submission

local_test = False

X_raw = pd.read_csv(os.path.join(IN_PATH, 'train_values.csv'), index_col='building_id')
train_labels = pd.read_csv(os.path.join(IN_PATH, 'train_labels.csv'), index_col='building_id')
y = train_labels.values.ravel()

if local_test:
    X_train_raw, X_eval_raw, y_train, y_eval = train_test_split(X_raw, y, test_size=0.25, stratify=y, random_state=123)

    ppc = Preprocessor()
    X_train = ppc.process(X_train_raw, is_train=True)
    X_eval = ppc.process(X_eval_raw, is_train=False)
    print(f'X_train.shape: {X_train.shape}')

    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_eval = model.predict(X_eval)

    f1_train = f1_score(y_train, pred_train, average='micro')
    f1_eval = f1_score(y_eval, pred_eval, average='micro')
    print(f'f1_train, f1_eval: {(round(f1_train, 2), round(f1_eval, 2))}')
    print(f'confusion_matrix train:\n{confusion_matrix(y_train, pred_train)}')
    print(f'confusion_matrix eval:\n{confusion_matrix(y_eval, pred_eval)}')
    print(f'accuracy_score train: {round(accuracy_score(y_train, pred_train), 2)}')
    print(f'accuracy_score eval: {round(accuracy_score(y_eval, pred_eval), 2)}')

make_submission(X_raw, y)
