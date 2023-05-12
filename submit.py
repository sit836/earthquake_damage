import os

import pandas as pd
from sklearn.linear_model import LogisticRegression

from constants import IN_PATH, OUT_PATH
from preprocessor import Preprocessor


def make_submission(X_raw, y):
    X_test_raw = pd.read_csv(os.path.join(IN_PATH, 'test_values.csv'), index_col='building_id')

    ppc = Preprocessor()
    X = ppc.process(X_raw, is_train=True)
    X_test = ppc.process(X_test_raw, is_train=False)

    model = LogisticRegression(random_state=0)
    model.fit(X, y)
    pred_test = model.predict(X_test)

    submission_format = pd.read_csv(os.path.join(IN_PATH, 'submission_format.csv'), index_col='building_id')
    my_submission = pd.DataFrame(data=pred_test,
                                 columns=submission_format.columns,
                                 index=submission_format.index)
    my_submission.to_csv(os.path.join(OUT_PATH, 'submission.csv'))
