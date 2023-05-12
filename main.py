import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from constants import IN_PATH
from preprocessor import Preprocessor
from submit import make_submission


X_raw = pd.read_csv(os.path.join(IN_PATH, 'train_values.csv'), index_col='building_id')
train_labels = pd.read_csv(os.path.join(IN_PATH, 'train_labels.csv'), index_col='building_id')
print(f'X_raw.shape: {X_raw.shape}')

y = train_labels.values.ravel()
X_train_raw, X_eval_raw, y_train, y_eval = train_test_split(X_raw, y, test_size=0.25, stratify=y, random_state=123)

ppc = Preprocessor()
X_train = ppc.process(X_train_raw, is_train=True)
X_eval = ppc.process(X_eval_raw, is_train=False)

# naive_pred_train = 2 * np.ones(len(y_train))
# naive_pred_eval = 2 * np.ones(len(y_eval))
# naive_f1_train = f1_score(y_train, naive_pred_train, average='micro')
# naive_f1_eval = f1_score(y_eval, naive_pred_eval, average='micro')
# print(f'f1_train, f1_eval: {(round(naive_f1_train, 2), round(naive_f1_eval, 2))}')
# print(f'confusion_matrix train:\n{confusion_matrix(y_train, naive_pred_train)}')
# print(f'confusion_matrix eval:\n{confusion_matrix(y_eval, naive_pred_eval)}')

model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_eval = model.predict(X_eval)

f1_train = f1_score(y_train, pred_train, average='micro')
f1_eval = f1_score(y_eval, pred_eval, average='micro')
print(f'f1_train, f1_eval: {(round(f1_train, 2), round(f1_eval, 2))}')
print(f'confusion_matrix train:\n{confusion_matrix(y_train, pred_train)}')
print(f'confusion_matrix eval:\n{confusion_matrix(y_eval, pred_eval)}')

# make_submission(X_raw, y)
