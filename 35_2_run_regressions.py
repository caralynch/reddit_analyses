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

OUTDIR = "regression_outputs"
# PARAMS_DICT_INFILE = f"{OUTDIR}/input_params.p"
REGRESSION_INFILE = "regression_thread_data.p"
THREAD_INFILE = "clean_5_thread_data.p"
SUBREDDITS = ["books", "crypto", "conspiracy", "politics"]
REGRESSION_TYPES = ["logistic", "linear", "mnlogit"]

start_time = dt.now().strftime("%d_%m_%Y__%H_%M_%S")
LOGDIR = f"{OUTDIR}/logs_{start_time}"
OUTFILE = f"{OUTDIR}/regressions.p"


def run_regression(params_dict):
    try:
        name = f"{params_dict['name']}_{params_dict['regression_type']}"
        logger.info(f"  {name}")
        # logfile handler
        sub_f_handler = logging.FileHandler(f"{LOGDIR}/{name}.log")
        sub_f_handler.setLevel(logging.INFO)
        sub_f_handler.setFormatter(f_format)
        # add handlers to the logger
        handlers = {"info": sub_f_handler, "warnings": sub_f_handler}
        regression = RR(params_dict, log_handlers=handlers)
        regression.main()
    except Exception as e:
        logger.exception(f" Exception occurred in {name} process")
    finally:
        out_dict[name] = regression
        logger.info(f"  {name} FINISHED")


if __name__ == "__main__":
    # warnings.simplefilter("ignore")
    #warnings.filterwarnings("ignore", category=RuntimeWarning)
    start = dt.now()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # stream handler for logging
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    try:
        if not os.path.isdir(OUTDIR):
            os.mkdir(OUTDIR)
        if not os.path.isdir(LOGDIR):
            os.mkdir(LOGDIR)
    except Exception as e:
        logger.exception("Exception occured with out directories")
        raise e

    # logfile handler
    f_handler = logging.FileHandler(f"{LOGDIR}/main.log")
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    logger.info(f"Start time {start}, PID {os.getpid()}")
    out_dict = {}

    try:
        logger.info("## Reading in files ##")
        logger.info("   Regression infile")
        regression_df = pickle.load(open(REGRESSION_INFILE, "rb"))
        logger.info("   Thread infile")
        thread_df = pickle.load(open(THREAD_INFILE, "rb"))
        logger.info("## Finished reading in files ##")

        # params_dicts = pickle.load(open(PARAMS_DICT_INFILE, "rb"))

        # param_dict_list = []
        # # need to get each param dict out
        # for subreddit in params_dicts:
        #     for regtype in params_dicts[subreddit]:
        #         param_dict_list.append(params_dicts[subreddit][regtype])
    except Exception as e:
        logger.exception("Exception occurred during file read in")
        raise e

    try:
        logger.info("## Creating parameter dictionaries ##")
        param_dict_list = []
        for subreddit in SUBREDDITS:
            for regression_type in REGRESSION_TYPES:
                param_dict_list.append(
                    RR.create_param_dict(
                        subreddit,
                        regression_type,
                        regression_df[subreddit],
                        thread_df[subreddit],
                    )
                )
        logger.info("## Finished creating parameter dictionaries ##")
    except Exception as e:
        logger.exception(
            f"Exception occured when creating {subreddit} {regression_type} dictionary"
        )
        raise e

    try:
        logger.info("## Starting pool processes ##")
        with Pool() as p:
            p.map(run_regression, param_dict_list)

        logger.info("## Finished pool processes ##")
    except Exception as e:
        logger.exception("Exception occurred with pool process")

    try:
        logger.info("## Writing to pickle file ##")
        pickle.dump(out_dict, open(OUTFILE, "wb"))

    except Exception as e:
        logger.exception("Exception occurred with output")
        raise e

    logger.info(f"FINISHED at {dt.now()}, time taken {dt.now()-start}")
