import numpy as np
import pandas as pd
from sklearn import preprocessing

import category_encoders as ce

from constants import BINARY_COLS, MULTIARY_COLS, NUM_COLS, GEO_LVLS
from utils import reduce_mem_usage


class Preprocessor:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')
        self.te = ce.TargetEncoder(cols=GEO_LVLS)
        self.poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.is_train = None

    def _standardize(self, X):
        if self.is_train:
            X_std = self.scaler.fit_transform(X)
        else:
            X_std = self.scaler.transform(X)
        return X_std

    def _enc_cat(self, X, y):
        X_no_change = X[NUM_COLS + BINARY_COLS]
        if self.is_train:
            X_ohe = self.ohe.fit_transform(X[MULTIARY_COLS]).toarray()
            X_te = self.te.fit_transform(X[GEO_LVLS], y)
        else:
            X_ohe = self.ohe.transform(X[MULTIARY_COLS]).toarray()
            X_te = self.te.transform(X[GEO_LVLS])
        return pd.DataFrame(np.hstack((X_no_change, X_ohe, X_te)),
                            columns=NUM_COLS + BINARY_COLS + list(
                                self.ohe.get_feature_names_out(MULTIARY_COLS)) + GEO_LVLS)

    def _add_interactions(self, X):
        # X_no_change = X[GEO_LVLS + BINARY_COLS + NUM_COLS]
        # if self.is_train:
        #     X_poly = self.poly.fit_transform(X[BINARY_COLS + NUM_COLS])
        # else:
        #     X_poly = self.poly.transform(X[BINARY_COLS + NUM_COLS])
        # return np.hstack((X_no_change, X_poly))

        if self.is_train:
            X_poly = self.poly.fit_transform(X)
        else:
            X_poly = self.poly.transform(X)
        return X_poly

    def process(self, is_train, is_tree, X, y=None):
        self.is_train = is_train
        X_processed = reduce_mem_usage(X)

        X_processed = self._enc_cat(X_processed, y)
        print('_enc_cat done')

        if not is_tree:
            X_processed = self._add_interactions(X_processed)
            print('_add_interactions done')

            X_processed = self._standardize(X_processed)

        return X_processed
