import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn import preprocessing

from constants import BINARY_COLS, MULTIARY_COLS, NUM_COLS, GEO_LVLS, OHE_COLS, TE_COLS, JS_COLS, QE_COLS, COUNT_COLS
from utils import reduce_mem_usage


class Preprocessor:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')
        self.te = ce.TargetEncoder(cols=TE_COLS)
        self.js = ce.JamesSteinEncoder(cols=JS_COLS)
        self.qe_25 = ce.QuantileEncoder(cols=GEO_LVLS, quantile=0.25)
        self.qe_75 = ce.QuantileEncoder(cols=GEO_LVLS, quantile=0.75)
        self.ce = ce.CountEncoder(cols=GEO_LVLS)
        self.poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.is_train = None

    def _standardize(self, X):
        if self.is_train:
            X_std = self.scaler.fit_transform(X)
        else:
            X_std = self.scaler.transform(X)
        return X_std

    def _enc_cat(self, X, y):
        X_no_change = X[NUM_COLS + BINARY_COLS + [col for col in X.columns if 'by' in col]]
        if self.is_train:
            X_ohe = self.ohe.fit_transform(X[OHE_COLS]).toarray()
            X_te = self.te.fit_transform(X[TE_COLS], y)
            X_js = self.js.fit_transform(X[JS_COLS], y)
            X_qe_25 = self.qe_25.fit_transform(X[GEO_LVLS], y)
            X_qe_75 = self.qe_75.fit_transform(X[GEO_LVLS], y)
            X_ce = self.ce.fit_transform(X[GEO_LVLS], y)
        else:
            X_ohe = self.ohe.transform(X[OHE_COLS]).toarray()
            X_te = self.te.transform(X[TE_COLS])
            X_js = self.js.transform(X[JS_COLS])
            X_qe_25 = self.qe_25.transform(X[GEO_LVLS])
            X_qe_75 = self.qe_75.transform(X[GEO_LVLS])
            X_ce = self.ce.transform(X[GEO_LVLS], y)
        X_iqr = X_qe_75 - X_qe_25
        return pd.DataFrame(np.hstack((X_no_change, X_ohe, X_te, X_js, X_iqr, X_ce)),
                            columns=NUM_COLS + BINARY_COLS + [col for col in X.columns if 'by' in col]
                                    + list(self.ohe.get_feature_names_out(OHE_COLS))
                                    + TE_COLS
                                    + JS_COLS
                                    + [f'{col}_iqr' for col in QE_COLS]
                                    + [f'{col}_count' for col in COUNT_COLS]
                            )

    def _add_interactions(self, X):
        if self.is_train:
            X_poly = self.poly.fit_transform(X[NUM_COLS + BINARY_COLS + GEO_LVLS])
        else:
            X_poly = self.poly.transform(X[NUM_COLS + BINARY_COLS + GEO_LVLS])
        return np.hstack((X, X_poly))

    def _add_group_stat(self, X):
        cat_cols = MULTIARY_COLS
        quantiles = [0.01, 0.50, 0.99]
        df_res = X.copy()
        for cat_col in cat_cols:
            for q in quantiles:
                df_gp_stat = X[NUM_COLS + [cat_col]].groupby(cat_col, as_index=False).quantile(q)
                df_gp_stat = df_gp_stat.rename(
                    columns={col: f'{col}_by_{cat_col}_q{q}' for col in df_gp_stat.columns if col not in [cat_col]})
                df_res = df_res.merge(df_gp_stat, how='left', on=cat_col)
        return df_res

    def _create_features(self, X):
        X['volume'] = X['area_percentage'] * X['height_percentage']
        return X

    def process(self, is_train, is_tree, X, y=None):
        self.is_train = is_train
        X_processed = reduce_mem_usage(X)

        # X_processed = self._add_group_stat(X_processed)
        # X_processed = self._create_features(X_processed)

        X_processed = self._enc_cat(X_processed, y)
        print('_enc_cat done')

        if not is_tree:
            X_processed = self._add_interactions(X_processed)
            print('_add_interactions done')

            X_processed = self._standardize(X_processed)

        return X_processed
