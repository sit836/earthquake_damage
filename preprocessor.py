import numpy as np
import pandas as pd
from sklearn import preprocessing

from constants import CAT_COLS, NUM_COLS, GEO_LVL1_COLS
from utils import reduce_mem_usage


class Preprocessor:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
        self.poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.is_train = None

    def _standardize(self, X):
        if self.is_train:
            X_std = self.scaler.fit_transform(X)
        else:
            X_std = self.scaler.transform(X)
        return X_std

    def _enc_cat(self, X):
        X_num = X[NUM_COLS]
        if self.is_train:
            X_ohe = self.enc.fit_transform(X[CAT_COLS]).toarray()
        else:
            X_ohe = self.enc.transform(X[CAT_COLS]).toarray()
        return pd.DataFrame(np.hstack((X_num, X_ohe)),
                            columns=NUM_COLS + list(self.enc.get_feature_names_out(CAT_COLS)))

    def _create_features(self, X):
        X_lvl_1 = X[GEO_LVL1_COLS]
        cols = [col for col in X.columns if col not in GEO_LVL1_COLS + NUM_COLS]
        if self.is_train:
            X_poly= self.poly.fit_transform(X[cols])
        else:
            X_poly = self.poly.transform(X[cols])
        return np.hstack((X_lvl_1, X_poly))

    def process(self, X, is_train):
        self.is_train = is_train
        X_processed = reduce_mem_usage(X)
        X_processed = self._enc_cat(X_processed)
        X_processed = self._create_features(X_processed)
        X_processed = self._standardize(X_processed)
        return X_processed
