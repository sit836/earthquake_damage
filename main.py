import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from constants import IN_PATH

train_values = pd.read_csv(os.path.join(IN_PATH, 'train_values.csv'), index_col='building_id')
train_labels = pd.read_csv(os.path.join(IN_PATH, 'train_labels.csv'), index_col='building_id')
print(f'train_values.shape: {train_values.shape}')

X = train_values
y = train_labels.values.ravel()
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, stratify=y, random_state=123)

# selected_features = ['foundation_type',
#                      'area_percentage',
#                      'height_percentage',
#                      'count_floors_pre_eq',
#                      'land_surface_condition',
#                      'has_superstructure_cement_mortar_stone']
scaler = preprocessing.StandardScaler()
oe = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# X_train = X_train[selected_features]
# X_eval = X_eval[selected_features]
cat_cols = list(X_train.select_dtypes(include='object').columns)

X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
X_eval[cat_cols] = oe.transform(X_eval[cat_cols])

X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)

###
naive_pred_train = 2 * np.ones(len(y_train))
naive_pred_eval = 2 * np.ones(len(y_eval))
naive_f1_train = f1_score(y_train, naive_pred_train, average='micro')
naive_f1_eval = f1_score(y_eval, naive_pred_eval, average='micro')
print(f'f1_train, f1_eval: {(round(naive_f1_train, 2), round(naive_f1_eval, 2))}')
print(f'confusion_matrix train:\n{confusion_matrix(y_train, naive_pred_train)}')
print(f'confusion_matrix eval:\n{confusion_matrix(y_eval, naive_pred_eval)}')

model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_eval = model.predict(X_eval)

f1_train = f1_score(y_train, pred_train, average='micro')
f1_eval = f1_score(y_eval, pred_eval, average='micro')
print(f'f1_train, f1_eval: {(round(f1_train, 2), round(f1_eval, 2))}')
print(f'confusion_matrix train:\n{confusion_matrix(y_train, pred_train)}')
print(f'confusion_matrix eval:\n{confusion_matrix(y_eval, pred_eval)}')
