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

# PARAMS_DICT_INFILE = f"{OUTDIR}/input_params.p"
REGRESSION_INFILE = "regression_thread_data.p"
THREAD_INFILE = "clean_5_thread_data.p"
SUBREDDITS = ["books", "crypto", "conspiracy", "politics"]
REGRESSION_TYPES = ["logistic", "linear", "mnlogit"]

start_time = dt.now().strftime("%d_%m_%Y__%H_%M_%S")
OUTDIR = f"regression_outputs/{start_time}"
LOGDIR = f"{OUTDIR}/logs"
RESULTSDIR = f"{OUTDIR}/results"
#OUTFILE = f"{OUTDIR}/regressions.p"


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
        regmod = RR(params_dict, log_handlers=handlers)
        regmod.main()
    except Exception as e:
        logger.exception(f" Exception occurred in {name} process")
    finally:
        try:
            logger.info(f"  Pickling {name} results")
            regmod.pickle_to_file(filename=f'{RESULTSDIR}/{name}.p')
        except Exception as e:
            logger.exception(f"  Exception occurred when pickling {name} results!")
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
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    try:
        for dirname in [OUTDIR, LOGDIR, RESULTSDIR]:
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
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

    logger.info(f"FINISHED at {dt.now()}, time taken {dt.now()-start}")
