import numpy as np
from sklearn import preprocessing

from constants import CAT_COLS


class Preprocessor:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
        self.is_train = None

    def _standardize(self, X):
        if self.is_train:
            X_std = self.scaler.fit_transform(X)
        else:
            X_std = self.scaler.transform(X)
        return X_std

    def _enc_cat(self, X):
        X_num = X.drop(columns=CAT_COLS).values

        if self.is_train:
            X_ohe = self.enc.fit_transform(X[CAT_COLS]).toarray()
        else:
            X_ohe = self.enc.transform(X[CAT_COLS]).toarray()
        return np.hstack((X_num, X_ohe))

    def process(self, X, is_train):
        self.is_train = is_train
        X_processed = self._enc_cat(X)
        X_processed = self._standardize(X_processed)
        return X_processed
