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

class RunRegressions:
    DEFAULT_PARAMS = {
        "subreddits": ["books", "crypto", "conspiracy"],
        "regression_types": ["logistic", "linear", "mnlogit"],
        "collection_window": 7,
        "model_window": 7,
        "x_cols": [
            "sentiment_sign",
            "sentiment_magnitude",
            "time_in_secs",
            "num_dayofweek",
            "activity_ratio",
            "mean_author_sentiment_sign",
            "mean_author_sentiment_magnitude",
            "author_all_activity_count",
            "domain_pagerank"
        ],
        "multiprocess": True,
    }

    REGRESSION_PARAMS = ["collection_window", "model_window", "x_cols"]

    def __init__(self, regression_df, thread_df, run_param_dict, out_dir):
        self.run_params = self.DEFAULT_PARAMS | run_param_dict
        if 'politics' in self.run_params['subreddits']:
            self.run_params['multiprocess'] = False
        self.out_dir = out_dir
        self.name = f"c{self.run_params['collection_window']}_m{self.run_params['model_window']}"

        # delete irrelevant data
        to_del = [x for x in regression_df if x not in self.run_params['subreddits']]
        self.regression_df = regression_df.copy()
        self.thread_df = thread_df.copy()
        for key in to_del:
            del self.regression_df[key]
            del self.thread_df[key]

        self.log_dir = f"{out_dir}/logs"
        self.results_dir = f"{out_dir}/results"
        
        try:
            for dirname in [out_dir, self.log_dir, self.results_dir]:
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
        except Exception as e:
            print("Exception occured with out directories")
            raise e
        
        self.extra_params = {}
        for i in [x for x in self.run_params if x in self.REGRESSION_PARAMS]:
            self.extra_params[i] = self.run_params[i]

        self.set_up_loggers()
    
    def set_up_loggers(self):
        self.logger = logging.getLogger(f"{__name__}_{self.run_params['name']}")
        self.logger.setLevel(logging.INFO)

        # stream handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        c_handler.setFormatter(c_format)
        self.logger.addHandler(c_handler)

        # logfile
        main_logfile = f"{self.log_dir}/{self.name}_main.log"
        if os.path.isfile(main_logfile):
            main_logfile = f"{self.log_dir}/{self.name}_main_{dt.now()}.log"
        f_handler = logging.FileHandler(main_logfile)
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(f_format)
        self.logger.addHandler(f_handler)

        self.logger.info(f"Logger set up")

    def get_param_dict_list(self):
        self.logger.info("## Creating parameter dictionaries ##")
        self.param_dict_list = []
        for subreddit in self.run_params['subreddits']:
            for regression_type in self.run_params['regressions_types']:
                self.param_dict_list.append(
                    RR.create_param_dict(
                        subreddit,
                        regression_type,
                        self.regression_df[subreddit],
                        self.thread_df[subreddit],
                        **self.extra_params,
                    )
                )
        # delete dicts no longer needed
        del self.regression_df
        del self.thread_df
        # collect garbage
        gc.collect()
        self.logger.info("## Finished creating parameter dictionaries ##")

    def run_multiprocessing(self):
        self.get_param_dict_list()
        try:
            self.logger.info("## Starting pool processes ##")
            with Pool() as p:
                p.map(self.run_regression, param_dict_list)
            # delete param list as no longer need
            del param_dict_list
            gc.collect()
        except Exception as e:
            self.logger.exception("Exception occurred with pool process")
    
    def run_linear(self):
        for subreddits in self.run_params['subreddits']:
            self.logger.info(f"## {subreddit} ##")
            for regression_type in self.run_params['regression_types']:
                self.logger.info(f"  # {regression_type} #")
                # TODO add here
        

    def run_regression(self, params_dict):
        try:
            name = f"{params_dict['name']}_{params_dict['regression_type']}"
            if self.run_params['multiprocess']:
                self.logger.info(f"  {name}")
            # logfile handlers
            sub_f_handler = logging.FileHandler(f"{self.log_dir}/{self.name}_{name}.log")
            sub_f_handler.setLevel(logging.INFO)
            f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            sub_f_handler.setFormatter(f_format)
            if not self.run_params['multiprocess']:
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
                self.logger.info("       handlers added")
                self.logger.info("       creating RR instance")
            else:
                handlers = {"info": sub_f_handler, "warnings": sub_f_handler}
                
            regmod = RR(params_dict, log_handlers=handlers)

            if not self.run_params['multiprocess']:
                self.logger.info("       running main")
            regmod.main()
        except Exception as e:
            self.logger.exception(f" Exception occurred in {name} process")
        finally:
            try:
                self.logger.info(f"  Pickling {name} results")
                regmod.pickle_to_file(filename=f"{self.results_dir}/{name}.p")
            except Exception as e:
                self.logger.exception(f"  Exception occurred when pickling {name} results!")
            del regmod
            self.logger.info(f"  {name} FINISHED")
    

