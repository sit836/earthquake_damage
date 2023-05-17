import os

import pandas as pd
from sklearn.linear_model import LogisticRegression

from constants import IN_PATH, OUT_PATH
from preprocessor import Preprocessor


def make_submission(X_raw, y):
    X_test_raw = pd.read_csv(os.path.join(IN_PATH, 'test_values.csv'), index_col='building_id')

    ppc = Preprocessor()
    X = ppc.process(is_train=True, X=X_raw, y=y)
    X_test = ppc.process(is_train=False, X=X_test_raw, y=None)

    model = LogisticRegression(random_state=0)
    model.fit(X, y)
    pred_test = model.predict(X_test)

    submission_format = pd.read_csv(os.path.join(IN_PATH, 'submission_format.csv'), index_col='building_id')
    my_submission = pd.DataFrame(data=pred_test,
                                 columns=submission_format.columns,
                                 index=submission_format.index)
    my_submission.to_csv(os.path.join(OUT_PATH, 'submission.csv'))
