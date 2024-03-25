import pickle
import pandas as pd
import numpy as np
from regression_class import RedditRegression as RR
import logging

REGRESSION_INFILE = "regression_thread_data.p"
THREAD_INFILE = "clean_5_thread_data.p"
regression_df = pickle.load(open(REGRESSION_INFILE, 'rb'))
thread_df = pickle.load(open(THREAD_INFILE, 'rb'))

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
s_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(s_format)

param_dicts = {}
regressions = {}
for regtype in ['logistic', 'mnlogit']:
    print(f'## {regtype} ##')
    param_dicts[regtype] = RR.create_param_dict(
        'books', regtype, regression_df['books'], thread_df['books'])
    regressions[regtype] = RR(param_dicts[regtype], log_handlers=handler)
    print(f'    attempting first pickle')
    regressions[regtype].pickle_to_file(
        f'regression_test_outputs/books_{regtype}1.p'
        )
    print(f'    first pickle successful')
    print(f'    running main')
    regressions[regtype].main()
    print(f'    attempting second pickle')
    regressions[regtype].pickle_to_file(
        f'regression_test_outputs/books_{regtype}2.p'
        )
    print(f'done')