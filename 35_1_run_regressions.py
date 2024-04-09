# data saving imports
import pickle
import os
import sys
import gc

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
REGRESSION_INFILE = "regression_thread_data_april_2024.p"
THREAD_INFILE = "clean_5_thread_data.p"
REGRESSION_TYPES = ["logistic", "linear", "mnlogit"]


def get_inputs():
    if len(sys.argv) < 2:
        # SUBREDDITS = ["books", "crypto", "conspiracy"]
        SUBREDDITS = ["politics"]
        # SUBREDDITS = ['conspiracy', 'politics']
        # REGRESSION_TYPES = ["mnlogit"]

        COLLECTION_WINDOW = 7
        MODEL_WINDOW = 7
    else:
        COLLECTION_WINDOW = int(sys.argv[1])
        MODEL_WINDOW = int(sys.argv[2])
        SUBREDDITS = [x for x in sys.argv[3:]]
    return COLLECTION_WINDOW, MODEL_WINDOW, SUBREDDITS


start_time = dt.now().strftime("%d_%m_%Y__%H_%M_%S")
start_date = dt.now().strftime("%d_%m_%Y")


X_COLS = [
    "sentiment_sign",
    "sentiment_magnitude",
    "time_in_secs",
    "num_dayofweek",
    "activity_ratio",
    "mean_author_sentiment_sign",
    "mean_author_sentiment_magnitude",
    "author_all_activity_count",
    "domain_pagerank",
    "domain_count",
]


def run_regression(params_dict):
    try:
        name = f"{params_dict['name']}_{params_dict['regression_type']}"
        if MULTIPROCESS:
            logger.info(f"  {name}")
        # logfile handlers
        sub_f_handler = logging.FileHandler(f"{LOGDIR}/{name}.log")
        sub_f_handler.setLevel(logging.INFO)
        sub_f_handler.setFormatter(f_format)
        if not MULTIPROCESS:
            sub_c_handler = logging.StreamHandler()
            sub_c_handler.setLevel(logging.INFO)
            c_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            sub_c_handler.setFormatter(c_format)
            handlers = {
                "info": [sub_f_handler, sub_c_handler],
                "warnings": sub_f_handler,
            }
            logger.info("       handlers added")
            logger.info("       creating RR instance")
        else:
            handlers = {"info": sub_f_handler, "warnings": sub_f_handler}

        regmod = RR(params_dict, log_handlers=handlers)

        if not MULTIPROCESS:
            logger.info("       running main")
        regmod.main()
    except Exception as e:
        logger.exception(f" Exception occurred in {name} process")
    finally:
        try:
            logger.info(f"  Pickling {name} results")
            regmod.pickle_to_file(filename=f"{RESULTSDIR}/{name}.p")
        except Exception as e:
            logger.exception(f"  Exception occurred when pickling {name} results!")
        del regmod
        logger.info(f"  {name} FINISHED")


if __name__ == "__main__":
    # warnings.simplefilter("ignore")
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    start = dt.now()

    COLLECTION_WINDOW, MODEL_WINDOW, SUBREDDITS = get_inputs()

    if "politics" in SUBREDDITS:
        MULTIPROCESS = False
    else:
        MULTIPROCESS = True

    OUTDIR = f"regression_outputs/{start_date}_c{COLLECTION_WINDOW}_m{MODEL_WINDOW}"
    LOGDIR = f"{OUTDIR}/logs"
    RESULTSDIR = f"{OUTDIR}/results"

    extra_params = {
        "collection_window": COLLECTION_WINDOW,
        "model_window": MODEL_WINDOW,
        "x_cols": X_COLS,
    }

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
    main_logfile = f"{LOGDIR}/main.log"
    if os.path.isfile(main_logfile):
        main_logfile = f"{LOGDIR}/main_{start_time}.log"
    f_handler = logging.FileHandler(main_logfile)
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

    # want to manage memory effectively
    try:
        logger.info("## Deleting irrelevant data ##")
        to_del = [x for x in regression_df if x not in SUBREDDITS]
        for key in to_del:
            del regression_df[key]
            del thread_df[key]
        logger.info("## Collecting garbage ##")
        gc.collect()
    except:
        logger.exception("  Something went wrong with data deletion.")

    if MULTIPROCESS:
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
                            **extra_params,
                        )
                    )
            # delete dicts no longer needed
            del regression_df
            del thread_df
            # collect garbage
            gc.collect()
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
            # delete param list as no longer need
            del param_dict_list
            gc.collect()
        except Exception as e:
            logger.exception("Exception occurred with pool process")
    else:
        for subreddit in SUBREDDITS:
            logger.info(f"## {subreddit} ##")
            for regression_type in REGRESSION_TYPES:
                logger.info(f"  # {regression_type} #")
                try:
                    logger.info("       Creating parameter dictionary")
                    param_dict = RR.create_param_dict(
                        subreddit,
                        regression_type,
                        regression_df[subreddit],
                        thread_df[subreddit],
                        **extra_params,
                    )
                    logger.info("       Created parameter dictionary")
                    logger.info("       Running regression")
                    run_regression(param_dict)
                    logger.info("       Finished running regression")
                except Exception as e:
                    logger.exception(
                        f" Exception occured with {subreddit} {regression_type}"
                    )
            logger.info("   # Deleting irrelevant data #")
            del regression_df[subreddit]
            del thread_df[subreddit]
            logger.info("   # Collecting garbage #")
            gc.collect()

    logger.info(f"FINISHED at {dt.now()}, time taken {dt.now()-start}")
