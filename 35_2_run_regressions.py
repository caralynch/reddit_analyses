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
start_time = dt.now().strftime('%d_%m_%Y__%H_%M_%S')
LOGFILE = f"{OUTDIR}/{start_time}"
OUTFILE = f"{OUTDIR}/regressions.p"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# stream handler for logging
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)


# logfile handler
f_handler = logging.FileHandler(f"{LOGFILE}.log")
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
        logger.info(f"#### {name} ####")
        # logfile handler
        sub_f_handler = logging.FileHandler(f"{LOGFILE}_{name}.log")
        sub_f_handler.setLevel(logging.INFO)
        sub_f_handler.setFormatter(f_format)
        # add handlers to the logger
        handlers = {
            'info': sub_f_handler,
            'warnings': sub_f_handler
        }
        regression = RR(params_dict, log_handlers=handlers)
        regression.main()
    except Exception as e:
        logger.exception(f"Exception occurred in {name} process")
    finally:
        out_dict[name] = regression
        logger.info(f"#### {name} FINISHED ####")


if __name__ == "__main__":
    #warnings.simplefilter("ignore")
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
    
    logger.info(f"FINISHED at {dt.now()}, time taken {dt.now()-start}")
