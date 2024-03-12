# data saving imports
import pickle
import os
import sys

# time tracking
from datetime import datetime as dt

# warnings
import warnings

# logging
import logging

# multiprocessing
from multiprocessing import Pool

# custom imports
from regression_class import RedditRegression as RR

OUTDIR = "regression_test_outputs"
PARAMS_DICT_INFILE = f"{OUTDIR}/input_params.p"
LOGFILE = f"{OUTDIR}/log{os.getpid()}.txt"
OUTFILE = f"{OUTDIR}/regressions.p"


logger = logging.getLogger(__name__)


# stream handler for logging
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)


# logfile handler
f_handler = logging.FileHandler(LOGFILE)
f_handler.setLevel(logging.INFO)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)

# add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

out_dict = {}


def run_regression(params_dict):
    try:
        name = f"{params_dict['name']}_{params_dict['regression_type']}"
        regression = RR(params_dict, logger=logger)
        regression.main()
        out_dict[name] = regression
    except Exception as e:
        out_dict[name] = regression
        logger.exception("Exception occurred")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    start = dt.now()
    logger.info(f"Start time {start}, PID {os.getpid()}")
    try:
        if not os.path.isdir(OUTDIR):
            os.mkdir(OUTDIR)
    except Exception as e:
        logger.exception("Exception occured with out directory")
        raise e

    try:
        logger.info("Reading in file")
        params_dicts = pickle.load(open(PARAMS_DICT_INFILE, "rb"))

        param_dict_list = []
        # need to get each param dict out
        for subreddit in params_dicts:
            for regtype in params_dicts[subreddit]:
                param_dict_list.append(params_dicts[subreddit][regtype])
    except Exception as e:
        logger.exception("Exception occurred during file read in")
        raise e

    try:
        logger.info("Starting pool processes")
        with Pool() as p:
            p.map(run_regression, param_dict_list)
    except Exception as e:
        logger.exception("Exception occurred with pool process")

    try:
        logger.info("Writing to pickle file")
        pickle.dump(out_dict, open(OUTFILE, "wb"))
    except Exception as e:
        logger.exception("Exception occurred with output")
        raise e
