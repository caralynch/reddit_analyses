# data types
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# for warnings, errors and exceptions management
import warnings
from numpy.linalg import LinAlgError
import statsmodels.tools.sm_exceptions as sme

# I/O
import pickle
import logging


# plots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# stats
from sklearn import metrics
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn import linear_model
from itertools import groupby

# for feature selection
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

"""
### TODO ###
Add thread size thresholding
Add domain tracker - e.g. domain frequency or something?
"""


class TimestampClass:
    """Class for various timestamp functions
    """

    @staticmethod
    def get_date(timestamp):
        """Get date from timestamp

        Parameters
        ----------
        timestamp : pd.Timestamp

        Returns
        -------
        pd.TimeStamp.date
        """
        return timestamp.date()

    @staticmethod
    def get_float_seconds(timestamp):
        hours = timestamp.hour
        minutes = timestamp.minute
        seconds = timestamp.second
        return hours * 60 * 60 + minutes * 60 + seconds

    @staticmethod
    def get_dayofweek(timestamp):
        return timestamp.dayofweek

    @staticmethod
    def get_hour(timestamp):
        return timestamp.hour
    
    @staticmethod
    def get_weekend_or_weekday(timestamp):
        weekday = timestamp.dayofweek
        if weekday < 5:
            return "Weekday"
        else:
            return "Weekend"
    
    @staticmethod
    def get_time_of_day(timestamp):
        h = timestamp.hour
        if 4 <= h <= 11:
            return "Morning"
        elif 12 <= h <= 19:
            return "Afternoon"
        else:
            return "Night"
        



class QuantileClass:
    """Class for segmenting data into quantiles
    """

    def __init__(self, data_series, quantiles):
        self.data_series = data_series
        self.quantiles = quantiles
        self.quantile_values = self.get_quantiles(data_series, quantiles)

    def main(self):
        self.get_range_tuples_()
        self.classify_cols_by_quantile_()
        out_dict = {
            "quantile_index_col": self.quantile_index_series,
            "quantile_ranges": self.range_tuples,
            "quantile_counts": self.values_per_quantile,
        }
        return out_dict

    def get_range_tuples_(self, weight_lower=False):
        self.weight_lower = weight_lower
        self.values_per_quantile = self.get_number_per_quantile(
            self.data_series, self.quantile_values, weight_lower=weight_lower
        )
        self.range_tuples = self.get_range_tuples(
            self.data_series, self.quantile_values, weight_lower=weight_lower
        )

    def classify_cols_by_quantile_(self):
        self.quantile_index_series = self.data_series.apply(
            self.find_quantile, range_tuples=self.range_tuples
        )
        self.quantile_range_series = self.data_series.apply(
            self.find_quantile, range_tuples=self.range_tuples, index=False
        )

    @staticmethod
    def get_quantiles(data_series, quantile_list):
        """Output list of quantile values.

        Parameters
        ----------
        data_series : pd.Series
            Series (column of df) of data
        quantile_list : list
            List of quantiles (e.g. [0.25, 0.5, 0.75])

        Returns
        -------
        list
            Values of each quantile
        """
        quantile_values = []
        for i in quantile_list:
            quantile_values.append(data_series.quantile(q=i))
        return quantile_values

    @staticmethod
    def get_number_in_range(data_series: pd.Series, range_tuple: tuple):
        """Get number of data points in series within given range, a closed interval.

        Parameters
        ----------
        data_series : pd.Series
            Data
        range_tuple : tuple
            Range of values to consider - considered a closed interval.

        Returns
        -------
        int
            Number of entries in data_series within the given range
        """
        data_in_range = data_series[
            (data_series >= range_tuple[0]) & (data_series <= range_tuple[1])
        ]
        return len(data_in_range)

    @classmethod
    def get_number_per_quantile(
        cls, data_series: pd.Series, quantile_values: list, weight_lower=False
    ):
        """Gives number of data entries for each given quantile. 

        Parameters
        ----------
        data_series : pd.Series
            Data for quantiles
        quantile_values : list
            List of values of quantiles
        weight_lower : bool
            Dictates whether range intervals are open or closed
        

        Returns
        -------
        pd.DataFrame
            DataFrame with range tuples as index and number of entries as column.
        """

        range_list = cls.get_range_tuples(data_series, quantile_values, weight_lower)
        ranges_dict = {}
        for range_tuple in range_list:
            ranges_dict[range_tuple] = cls.get_number_in_range(data_series, range_tuple)
        ranges_df = pd.DataFrame.from_dict(
            ranges_dict, orient="index", columns=["count"]
        )
        ranges_df.index.name = "range"
        return ranges_df

    @staticmethod
    def get_range_tuples(
        data_series: pd.Series, quantile_values: list, weight_lower=False
    ):
        quantile_values = [data_series.min()] + quantile_values + [data_series.max()]
        range_list = []
        for i in range(0, len(quantile_values) - 1):
            if not weight_lower:
                lower = quantile_values[i]
                if (i + 2) == len(quantile_values):
                    upper = quantile_values[i + 1]
                else:
                    upper = data_series[data_series < quantile_values[i + 1]].max()
            else:
                if i == 0:
                    lower = quantile_values[i]
                else:
                    lower = data_series[data_series > quantile_values[i]].min()
                upper = quantile_values[i + 1]
            range_list.append((lower, upper))
        return range_list

    @staticmethod
    def find_quantile(value, range_tuples, index=True):
        for i, range_tuple in enumerate(range_tuples):
            if range_tuple[0] <= value <= range_tuple[1]:
                if index:
                    return i
                else:
                    return range_tuple


class RedditRegression(TimestampClass, QuantileClass):
    
    

    ## CLASS LOOKUPS
    COLUMN_FUNCTIONS = {
        "time_in_secs": TimestampClass.get_float_seconds,
        "num_dayofweek": TimestampClass.get_dayofweek,
        "hour": TimestampClass.get_hour,
        "weekday": TimestampClass.get_weekend_or_weekday,
        "time_of_day": TimestampClass.get_time_of_day
    }

    WEEKDAY_MAP = {
        "Weekday": 0,
        "Weekend": 1,
    }

    DAYTIME_MAP = {
        "Morning": 0,
        "Afternoon": 1,
        "Night": -1,
    }

    SMF_FUNCTIONS = {"logistic": "logit", "linear": "ols", "mnlogit": "mnlogit"}

    SKL_FUNCTIONS = {
        "logistic": linear_model.LogisticRegression(),
        "linear": linear_model.LinearRegression(),
        "mnlogit": linear_model.LogisticRegression(multi_class="multinomial"),
    }

    SMF_PARAMS_LOOKUP = {"aic": "aic", "bic": "bic", "r2": "rsquared"}

    ## DEFAULT VALUES
    DEFAULT_Y_COL = {
        "logistic": "success",
        "linear": "thread_size",
        "mnlogit": "thread_size",
    }

    ## PRETTY FEATURE NAMES
    FEATURE_NAME_LOOKUP = {
    'domain_count': 'Post domain count',
    'author_all_activity_count': 'Author activity count',
    'mean_author_sentiment_magnitude': 'Author mean sentiment magnitude',
    'mean_author_sentiment_sign': 'Author mean sentiment sign',
    'domain_pagerank': 'Post domain PageRank',
    'activity_ratio': 'Author activity ratio',
    'time_in_secs': 'Time of day',
    'sentiment_sign': 'Post sentiment sign',
    'sentiment_magnitude': 'Post sentiment magnitude',
    'num_dayofweek': 'Day of week'
}

    ## INITIALISE CLASS
    def __init__(self, regression_params: dict, log_handlers=None):
        """Initialise regression data class

        Parameters
        ----------
        regression_params : dict
            Dictionary with:
            'name': ideally subreddit name
            'regression_data': df of regression data,
            'thread_data': df of thread data,
            'regression_type': str indicating whether logistic, linear or multinomial logistic regression
            'collection_window': size of data collection window (used to calc author
                stats), in days,
            'model_window': size of modelling window in days,
            'validation_window': size of validation window in days,
            'FSS': whether to do FSS or not,
            'performance_scoring_method': method used to score FSS,
            'x_cols': features to model (used when doing FSS),
            'y_col': success column (used when doing FSS),
            'quantiles': list of quantiles to use for mnlogit, optional
            'models': models to use (used when not doing FSS)
            'metrics': list of metrics to output (AUC, AIC,...)
            'thresholds': dict of thresholds to include on model data, in the format key = column name to be thresholded and value = threshold value, with model data filtered such that only rows above or including that value are included.
            'scale': True if wish to use sklearn's scaler in preprocessing.
        log_handlers : list, optional
            list of log handlers to pass to class. Defaults to empty list and standard
            streamhandler is set up.
        """
        # to avoid warning dating when creating new cols
        pd.options.mode.chained_assignment = None

        # set up logging
        self.set_up_loggers(
            log_handlers,
            name=f"{regression_params['name']}_{regression_params['regression_type']}",
        )

        self.PERFORMANCE_SCORING_METHODS = {
            "logistic": "roc_auc",
            "linear": "r2",
            "mnlogit": self.mnlogit_accuracy_score,
        }

        regression_params = regression_params.copy()

        self.regression_data = regression_params["regression_data"]
        del regression_params["regression_data"]

        # want sentiment in thread data to be one col
        regression_params["thread_data"].loc[:, "sentiment_score"] = regression_params[
            "thread_data"
        ].apply(self.get_score, axis=1)

        # only need certain cols from thread data
        self.thread_data = regression_params["thread_data"][
            ["thread_id", "id", "timestamp", "author", "score", "sentiment_score", "domain"]
        ]

        del regression_params["thread_data"]

        # construct regression params dict
        self.construct_regression_params_dict(regression_params)

        self.manage_dates_and_windows()

        # create dicts for info collection
        self.regression_metrics = {}
        self.model_data = {}
        self.num_threads_modelled = {}

        # back to warning
        pd.options.mode.chained_assignment = "warn"

    def set_up_loggers(self, log_handlers, name):
        """Sets up the loggers for the class.

        Parameters
        ----------
        log_handlers : list, dict or logging.handler
            A list, dict or logging.handler to pass to the class loggers.
        name : str
            Name of info logger
        """
        # set up logging
        logging.captureWarnings(True)

        self.loggers = {
            "warnings": logging.getLogger("py.warnings"),
            "info": logging.getLogger(f"{__name__}_{name}"),
        }
        self.loggers["info"].setLevel(logging.DEBUG)

        if log_handlers is not None:
            if isinstance(log_handlers, list):
                for handler in log_handlers:
                    for key in self.loggers:
                        self.loggers[key].addHandler(handler)

            elif isinstance(log_handlers, dict):
                for key in log_handlers:
                    if isinstance(log_handlers[key], list):
                        for handler in log_handlers[key]:
                            self.loggers[key].addHandler(handler)
                    else:
                        self.loggers[key].addHandler(log_handlers[key])
            else:
                for key in self.loggers:
                    self.loggers[key].addHandler(log_handlers)
        else:
            # stream handler for logging
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            s_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(s_format)
            self.loggers["info"].addHandler(handler)

    def construct_regression_params_dict(self, regression_params):
        """ Makes dict to keep track of regression params.
        """
        self.regression_params = {}

        defaults = {
            "name": "",
            "regression_type": "logistic",
            "metrics": ["roc_auc"],
            "scale": True,
        }

        for key in regression_params:
            self.regression_params[key] = regression_params[key]

        for key in [x for x in defaults if x not in regression_params]:
            self.regression_params[key] = defaults[key]

        self.add_params_for_FSS_or_models()

        # add default quantiles for mnlogit data
        if (
            self.regression_params["regression_type"] == "mnlogit"
            and "quantiles" not in self.regression_params
        ):
            self.regression_params["quantiles"] = [0.25, 0.5, 0.75]

        # author activity threshold
        if "thresholds" in self.regression_params:
            self.removed_threads = {}

    def add_params_for_FSS_or_models(self):
        """Adds required parameters and empty dicts for FSS true or false.
        """

        if "FSS" in self.regression_params and (self.regression_params["FSS"] == True):
            # if doing FSS, then need to create empty dicts for FSS metrics
            self.FSS_metrics = {}
            self.FSS_metrics_df = {}

            # also need to make sure there's a performance scoring method
            if "performance_scoring_method" not in self.regression_params:
                self.regression_params[
                    "performance_scoring_method"
                ] = self.PERFORMANCE_SCORING_METHODS[
                    self.regression_params["regression_type"]
                ]
            elif "mnlogit" in self.regression_params["performance_scoring_method"]:
                self.regression_params[
                    "performance_scoring_method"
                ] = self.PERFORMANCE_SCORING_METHODS["mnlogit"]
                self.regression_params[
                    "performance_scoring_method_str"
                ] = "accuracy_score"

            # there also needs to be x and y columns
            if "y_col" not in self.regression_params:
                self.regression_params["y_col"] = self.DEFAULT_Y_COL[
                    self.regression_params["regression_type"]
                ]
        else:
            # if not performing FSS, should have been given the models to run
            self.regression_params["x_cols"] = self.get_x_vals_from_modstring_dict(
                self.regression_params["models"]
            )
            # determine y vals
            y_cols = list(
                set(
                    [
                        self.get_y_from_modstring(i)
                        for i in list(self.regression_params["models"].values())
                    ]
                )
            )
            if len(y_cols) == 1:
                self.regression_params["y_col"] = y_cols[0]
            else:
                self.regression_params["y_col"] = y_cols

    def manage_dates_and_windows(self):
        """Create the date array and create model, collection and validation windows if
        not given
        """
        # get array of dates in dataset
        date_array = self.thread_data.timestamp.apply(TimestampClass.get_date).unique()

        # the dataset may be missing days - so create df from start and end dates
        date_array = pd.date_range(start=date_array[0], end=date_array[-1])
        self.date_array = pd.DataFrame(date_array)[0].apply(TimestampClass.get_date)

        # check all required windows are present
        # expected portions
        windows = {
            "collection_window": 1 / 4,
            "model_window": 1 / 2,
            "validation_window": 1 / 4,
        }

        # if the set difference is empty, then all windows present, if will not occur
        missing_windows = set(windows.keys()) - set(self.regression_params.keys())
        if missing_windows:
            present_windows = set(windows.keys()) & set(self.regression_params.keys())
            taken_days = 0
            for key in present_windows:
                taken_days += self.regression_params[key]
            days_left = len(self.date_array) - taken_days
            if days_left <= 0:
                for key in missing_windows:
                    self.regression_params[key] = 0
            elif days_left == 1:
                if "collection_window" in missing_windows:
                    self.regression_params["collection_window"] = days_left
                elif "model_window" in missing_windows:
                    self.regression_params["model_window"] = days_left
                else:
                    self.regression_params["validation_window"] = days_left
            else:
                missing_windows_list = [x for x in missing_windows]
                missing_windows_proportions = [windows[x] for x in missing_windows_list]
                new_missing_windows_proportions = [
                    x * sum(missing_windows_proportions)
                    for x in missing_windows_proportions
                ]
                for i, key in enumerate(missing_windows_list):
                    self.regression_params[key] = round(
                        new_missing_windows_proportions[i] * days_left
                    )
        else:
            # if not missing windows, then can use window size to determine date array size
            total_window_size = 0
            for w in windows:
                total_window_size += self.regression_params[w]
            # update so that date array only includes required dates for given windows
            self.date_array = self.date_array[:total_window_size]

    def calc_collection_counts(self):
        """Update thread data df with author activity, post and comment counts. These
        are calculated by looking at the collection window preceding the day of current
        activity.

        E.g. If author A has posted 5 times between days 6-10, and the collection window
        is 5 days, then on day 11 the collection window is days 6-10 and the activity
        count will be 5.
        """
        # create new cols in thread data df to store author data
        new_cols = [
            "author_all_activity_count",
            "author_post_count",
            "author_comment_count",
            "activity_ratio",
            "mean_author_sentiment",
            "mean_author_score",
        ]
        # NOTE is creating these regardless of desired x_cols innefficient??

        # establish condition for getting domain data
        domain_condition = ("domain_count" in self.regression_params["x_cols"]) | (
            "log_domain_count" in self.regression_params["x_cols"]
        )

        if domain_condition:
            new_cols.append("domain_count")

        # to avoid warning dating when creating new cols
        pd.options.mode.chained_assignment = None
        for new_col in new_cols:
            self.thread_data[new_col] = 0

        # need to go through each day after initial collection window up to end of data
        for day in range(
            self.regression_params["collection_window"], len(self.date_array)
        ):

            # collection window is current day - collection window, up to (and
            # excluding) current day
            collection_window = self.date_array[
                day - self.regression_params["collection_window"] : day
            ]

            # find thread data in collection window
            collection_thread_data = self.thread_data[
                self.thread_data.timestamp.apply(TimestampClass.get_date).isin(
                    collection_window
                )
            ]

            # find thread data on current day
            day_thread_data = self.thread_data[
                self.thread_data.timestamp.apply(TimestampClass.get_date)
                == self.date_array[day]
            ]

            # only need to look at collection data for authors that were active today
            # if looking at domains, then also need domain counts
            collection_condition = collection_thread_data.author.isin(
                day_thread_data.author
            )
            if domain_condition:
                collection_condition = (
                    collection_condition
                    | collection_thread_data.domain.isin(day_thread_data.domain)
                )

            collection_thread_data = collection_thread_data[collection_condition]

            if domain_condition:
                domain_count = (
                    collection_thread_data[["id", "domain"]]
                    .groupby("domain")
                    .count()
                    .rename(columns={"id": f"domain_count"})
                )

            # separate by activity
            thread_activity = {
                "all_activity": collection_thread_data,
                "post": collection_thread_data[
                    collection_thread_data.thread_id == collection_thread_data.id
                ],
                "comment": collection_thread_data[
                    collection_thread_data.thread_id != collection_thread_data.id
                ],
            }

            started = False
            for key in thread_activity:
                author_activity_count = (
                    thread_activity[key][["author", "id"]]
                    .groupby("author")
                    .count()
                    .rename(columns={"id": f"author_{key}_count"})
                )
                if not started:
                    author_activity = author_activity_count
                    started = True
                else:
                    author_activity = (
                        pd.concat((author_activity, author_activity_count), axis=1)
                        .fillna(0)
                        .astype(int)
                    )

            # get activity ratio
            author_activity["activity_ratio"] = (
                author_activity.author_comment_count - author_activity.author_post_count
            ) / author_activity.author_all_activity_count

            # get mean sentiment score
            mean_author_scores_lookup = {
                "mean_author_sentiment": "sentiment_score",
                "mean_author_score": "score",
            }

            author_info = author_activity
            for key in mean_author_scores_lookup:
                new_author_col = thread_activity["all_activity"][["author", mean_author_scores_lookup[key]]].groupby("author").mean().rename(columns={mean_author_scores_lookup[key]: key})
                author_info = pd.concat((author_info, new_author_col), axis=1)


            # convert to dict of mapping dicts to add to thread data from day
            author_info_maps = author_info.to_dict()
            if domain_condition:
                domain_maps = domain_count.to_dict()["domain_count"]
                day_thread_data["domain_count"] = day_thread_data.domain.map(
                    domain_maps
                ).fillna(0)
                self.thread_data.loc[
                    day_thread_data.index, "domain_count"
                ] = day_thread_data["domain_count"]

            # map author info to authors in today's thread data and update thread data
            # with today's author colleciton data
            for new_col in author_info_maps:
                day_thread_data[new_col] = day_thread_data.author.map(
                    author_info_maps[new_col]
                ).fillna(0)
                self.thread_data.loc[day_thread_data.index, new_col] = day_thread_data[
                    new_col
                ]

        # get log of author activity if included
        cols = ["log_author_all_activity_count", "log_domain_count"]
        for col in [x for x in cols if x in self.regression_params["x_cols"]]:
            self.thread_data[col] = np.log(
                self.thread_data.loc[:, col.removeprefix("log_")] + 1
            )

        # separate mean author sentiment into mag and sign
        col = "mean_author_sentiment"
        if col in self.thread_data.columns:
            (
                self.thread_data[f"{col}_sign"],
                self.thread_data[f"{col}_magnitude"],
            ) = self.separate_float_into_sign_mag(self.thread_data[col])

        # back to warning
        pd.options.mode.chained_assignment = "warn"

    def get_cal_val_data(self):
        """Performs all necessary operations to create the dfs for modelling and
        validation.
        """

        # get thread author data
        self.calc_collection_counts()

        # get model data
        self.get_regression_model_data()

        # get validation data
        if "validation_window" in self.regression_params:
            self.get_regression_model_data(calval="val")

        # if scaling, perform now
        if self.regression_params["scale"]:
            self.perform_scaling()
        else:
            self.__model_data__ = self.model_data

    def main(self):
        """
        Calculates author count data from thread data with a rolling collection window.
        - gets regression model data
        - runs FSS if required and gets sm modstrings
        - iterates through every proposed model:
            - runs logistic regressions
            - calculates metrics required (AIC, BIC, AUC)
        """
        # to avoid warning dating when creating new cols
        pd.options.mode.chained_assignment = None
        # get calibration and validation dfs
        self.get_cal_val_data()

        # if need FSS, run FSS
        if "FSS" in self.regression_params:
            self.loggers["info"].info("Running FSS")
            self.sm_modstrings = self.run_FSS()
            self.loggers["info"].info("FSS finished")
        else:
            self.sm_modstrings = self.regression_params["models"]

        # run models
        self.loggers["info"].info("Running models")
        self.run_models()

        # back to warning
        pd.options.mode.chained_assignment = "warn"

    def run_models(self):
        """Run models from obtained modstrings. Save metrics to regression_metrics dict.
        """

        # iterate through each model
        model_results = {}
        param_dict = {}
        self.smf_models = {}
        for mod_key in self.sm_modstrings:
            self.loggers["info"].info(f"Model {mod_key}")
            self.loggers["info"].debug("    Running regression")
            self.smf_models[mod_key] = self.run_regression(mod_key)
            self.loggers["info"].debug("    Getting regression metrics")
            model_results[mod_key] = self.get_regression_metrics(
                self.smf_models[mod_key], mod_key
            )

            param_dict[mod_key] = self.get_model_metrics_from_smf_mod(
                self.smf_models[mod_key]
            )

        # convert model metrics dict to df
        model_results = pd.DataFrame.from_dict(model_results, orient="index")

        self.regression_metrics = {
            "regression_params": param_dict,
            "metrics": model_results,
        }

    def pickle_to_file(self, filename: str):
        """Saves class instance to pickle.

        Parameters
        ----------
        filename : str
            file name for pickle file
        """
        pickle.dump(self, open(filename, "wb"))

    def get_model_metrics_from_smf_mod(self, smf_mod, conf_int=0.05):
        """Creates parameter dataframe of model metrics for easy output.

        Parameters
        ----------
        smf_mod : smf.ModelResults
            smf ModelResults instance.
        conf_int : float, optional
            Confidence interval alpha, by default 0.05

        Returns
        -------
        pd.DataFrame
            Dataframe with regression parameters, p values and stderr or conf interval.
        """

        if self.regression_params["regression_type"] != "mnlogit":
            params_df = pd.DataFrame([smf_mod.params, smf_mod.pvalues]).T.rename(
                columns={0: "param", 1: "pvalue"}
            )

            conf_df = smf_mod.conf_int(alpha=conf_int).rename(
                columns={0: "conf_low", 1: "conf_high"}
            )

            params_df = pd.concat((params_df, conf_df), axis=1)

        else:
            lookup = {"p_value": smf_mod.pvalues.T, "param": smf_mod.params.T}

            params_df = smf_mod.conf_int(alpha=conf_int).copy()
            params_df.rename(
                columns={"lower": "conf_low", "upper": "conf_high"}, inplace=True
            )

            for key in lookup:
                params_df[key] = ""

            for index_tuple in params_df.index:
                i = index_tuple[0]
                val = index_tuple[1]
                for key in lookup:
                    params_df.loc[(i, val), key] = lookup[key].loc[int(i) - 1, val]

            params_df = params_df[["param", "p_value", "conf_low", "conf_high"]]

        return params_df

    def perform_scaling(self):
        """Scale x column data using standard sklean Standard Scaler
        """

        self.scale_metrics_dict = {}
        self.scaled_data = {}
        for calval in self.model_data:
            x_data = self.model_data[calval][self.regression_params["x_cols"]]

            # don't scale binary cols
            binary_cols = list(x_data.columns[x_data.isin([-1,0,1]).all()])
            x_to_scale = x_data[[i for i in x_data if i not in binary_cols]]
            x_not_to_scale = x_data[binary_cols]


            scaled_x_data, self.scale_metrics_dict[calval] = self.scale_data(x_to_scale)
            scaled_x_data = pd.concat((scaled_x_data, x_not_to_scale), axis=1)
            y_data = self.model_data[calval][[self.regression_params["y_col"]]]
            self.scaled_data[calval] = pd.concat((y_data, scaled_x_data), axis=1)
        self.__model_data__ = self.scaled_data

    @staticmethod
    def scale_data(x_data):
        scaler = preprocessing.StandardScaler().fit(x_data)
        scaler_info = {"mean": scaler.mean_, "scale": scaler.scale_}
        x_scaled = pd.DataFrame(
            scaler.transform(x_data), columns=x_data.columns, index=x_data.index
        )
        return x_scaled, scaler_info

    def get_regression_model_data(self, date_index=0, calval="cal"):
        """Get regression data for model window only, and merge with author data from
        collection thread data

        Parameters
        ----------
        date_index : int
            Index of start date of collection window
        calval: str
            Indicates whether the model data is for calibration (uses model window)
            or validation (uses validation window)
        
        Returns
        -------
        pd.DataFrame
            Regression data that is within model dates, with all required cols
        """
        start = date_index + self.regression_params["collection_window"]
        if calval == "cal":
            end = start + self.regression_params["model_window"]
        else:
            # if doing validation, validation window starts after model window
            start += self.regression_params["model_window"]
            end = start + self.regression_params["validation_window"]

        # find model window dates
        model_dates = self.date_array[start:end]
        model_data = self.regression_data[
            self.regression_data.timestamp.apply(TimestampClass.get_date).isin(
                model_dates
            )
        ]

        # combine author collection data with model data
        thread_data_cols = ["id"] + [
            x for x in self.thread_data.columns if x in self.regression_params["x_cols"]
        ]

        # need to add columns for filtering purposes if thresholding
        if "thresholds" in self.regression_params:
            for key in self.regression_params["thresholds"]:
                if (key not in thread_data_cols) & (key not in model_data.columns):
                    thread_data_cols += [key]

        # only if there are actually thread cols to add (so if considering author
        # features)
        if len(thread_data_cols) > 1:
            model_data = self.merge_author_and_model_data(model_data, thread_data_cols)

        # TODO DOMAIN INFO SHOULD PROBABLY HAPPEN HERE???

        # if there's an author activity threshold, perform thresholding
        if "thresholds" in self.regression_params:
            model_data = self.perform_thresholding(model_data, calval)

        # make other required cols
        
        for col in [
            x for x in self.COLUMN_FUNCTIONS if x in self.regression_params["x_cols"]
        ]:
            new_col = model_data.timestamp.apply(self.COLUMN_FUNCTIONS[col])
            if is_numeric_dtype(new_col):
                model_data[col] = new_col
            elif new_col.isin(self.WEEKDAY_MAP).all():
                model_data[col] = new_col.map(self.WEEKDAY_MAP)
            elif new_col.isin(self.DAYTIME_MAP).all():
                model_data[col] = new_col.map(self.DAYTIME_MAP)
            else:
                # for categorical data, need to get dummy columns and change the x cols
                new_col = new_col.astype("category")
                new_cols = new_col.str.get_dummies()
                model_data = pd.concat((model_data, new_cols), axis=1)

                # only change x cols etc if in val
                if ("validation_window" not in self.regression_params) | (("validation_window" in self.regression_params) and (calval == "val")):
                    self.regression_params["x_cols"].remove(col)
                    self.regression_params["x_cols"] += list(new_cols.columns)



        # if mnlogit, then need to classify y data
        if self.regression_params["regression_type"] == "mnlogit":
            model_data = self.get_y_col_quantiles(model_data, calval)
            self.get_quantile_metrics()

        if calval == "cal":
            self.regression_model_data = model_data
        else:
            self.validation_data = model_data

        self.model_data[calval] = model_data

    def get_y_col_quantiles(self, model_data, calval="cal"):
        """If this is a multinomial logistic regression, then the y column needs to be
        divided into classes - this is done with the model data, after threshold, to
        obtain roughly equally sized classes. This updates the model data df by
        classifying the y col into equal classes, creating a new column for it, and
        updating the y_col entry in the regression params dict.

        Parameters
        ----------
        model_data : pd.DataFrame
            Given model data (which includes necessary y col)
        calval : str
            Indicates whether this is calibration or validation data

        Returns
        -------
        pd.DataFrame
            Model data with an additional column corresponding to the new y column.
        """
        y_col = self.regression_params["y_col"]
        if calval == "cal":
            self.regression_params["input_y_col"] = y_col
            quantile_obj = QuantileClass(
                model_data[y_col], self.regression_params["quantiles"]
            )
            quantile_data = quantile_obj.main()
            y_col += "_quantile_index"
            self.regression_params["y_col"] = y_col
            model_data[y_col] = quantile_data["quantile_index_col"]
            del quantile_data["quantile_index_col"]
            self.quantile_data = quantile_data
        else:
            quantile_ranges = self.quantile_data["quantile_ranges"].copy()
            input_y_col = self.regression_params["input_y_col"]
            quantile_ranges, counts = self.check_data_in_ranges(
                model_data[input_y_col], quantile_ranges
            )
            self.quantile_data["val_quantile_ranges"] = quantile_ranges
            self.quantile_data["quantile_counts"]["val_count"] = counts
            model_data[y_col] = model_data[input_y_col].apply(
                QuantileClass.find_quantile, range_tuples=quantile_ranges
            )
        return model_data

    def get_quantile_metrics(self):
        """Creates a self.quantile_metrics df to store quantile metrics for easy output & readability.
        """
        ranges_dict = {
            x: self.quantile_data[x] for x in self.quantile_data if "ranges" in x
        }
        ranges_df = pd.DataFrame(ranges_dict)
        count_df = self.quantile_data["quantile_counts"].copy().reset_index(drop=True)
        self.quantile_metrics = pd.concat((ranges_df, count_df), axis=1)

    @staticmethod
    def check_data_in_ranges(data: pd.Series, range_tuples: list):
        """Checks the given data is in the given ranges, if not it extends the ranges
        by either lowering the start of the first interval or raising the end of the
        last interval, or both. Also calculates data counts in each interval.

        Parameters
        ----------
        data : pd.Series
            Data to segment.
        range_tuples : list
            List of tuples representing intervals.

        Returns
        -------
        list(tuple), list(int)
            Returns the updated list of range tuples and data points in each interval.
        """
        # if there are smaller numbers in the data range, then extend first bin
        if data.min() < range_tuples[0][0]:
            range_tuples[0] = (data.min(), range_tuples[0][1])
        # if there are larger numbers in the data range, then extend the last bin
        if data.max() > range_tuples[-1][1]:
            range_tuples[-1] = (range_tuples[-1][0], data.max())
        counts = []
        for range_tuple in range_tuples:
            counts.append(QuantileClass.get_number_in_range(data, range_tuple))

        return range_tuples, counts

    def perform_thresholding(self, model_data: pd.DataFrame, calval: str):
        """Performs all relevant thresholding on given model data. Also saves removed
        threads to removed_threads.

        Parameters
        ----------
        model_data : pd.DataFrame
            Model data to perform author activity thresholding on.
        calval : str
            Whether data is cal or val.

        Returns
        -------
        pd.DataFrame
            Model data with thresholding applied.
        """
        # prepare dict to save removed threads to
        self.removed_threads[calval] = {}

        # iterate through all thresholds given in thresholds dict
        for colname in self.regression_params["thresholds"]:
            threshold = self.regression_params["thresholds"][colname]

            # save threads that have this column value below threshold
            self.removed_threads[calval][colname] = model_data.loc[
                model_data[colname] < threshold, :
            ]
            # update model data to only include threads with column value above threshold
            model_data = model_data.loc[model_data[colname] >= threshold, :]

        # drop the columns not required for modelling that were kept only for thresholding
        to_drop = [
            col
            for col in self.regression_params["thresholds"]
            if (
                (col not in self.regression_params["x_cols"])
                & (col not in self.regression_params["y_col"])
            )
        ]
        if len(to_drop) > 0:
            model_data.drop(labels=to_drop, axis=1, inplace=True)

        return model_data

    def merge_author_and_model_data(
        self, model_data: pd.DataFrame, thread_data_cols: list
    ):
        """Merges author data calculated with thread_data, with the given model_data.
        Performs a left hand merge with model data, such that only threads are left.
        Only merges columns that are specified in the regression params.

        Parameters
        ----------
        model_data : pd.DataFrame
            regression data in model time window
        thread_data_cols : list
            list of thread data columns to merge with regression data
        calval : str
            whether given data is calibration or validation data

        Returns
        -------
        pd.DataFrame
            model data with required author columns
        """
        model_data = model_data.merge(
            self.thread_data[thread_data_cols],
            left_on="thread_id",
            right_on="id",
            how="left",
        )

        model_data.drop(labels=["id"], axis=1, inplace=True)

        return model_data

    @staticmethod
    def manage_convergence_warnings(fit_function, **kwargs):
        """Checks whether the fit function runs and converges, if it doesn't suggests
        new parameters for next iteration by modifying the solver and max iterations.

        Parameters
        ----------
        fit_function : smf_model.fit
            Statsmodels.formula.api model.fit function.

        Returns
        -------
        bool, dict
            Bool indicating whether to run the function again or not, and dictionary of
            kwargs for fit function.
        """
        lookup_dict = {"method": "bfgs", "maxiter": 100}
        maxiter_limit = 1500
        run_again = False
        methods_list = ['bfgs', "cg", "nm", "newton", "lbfgs", "powell", "ncg"]
        with warnings.catch_warnings(record=True) as w:
            try:
                fit_function(disp=0, **kwargs)
            except (LinAlgError, sme.PerfectSeparationError) as e:
                if "method" not in kwargs or kwargs["method"] != "bfgs":
                    kwargs["method"] = "bfgs"
                elif kwargs["method"] == "bfgs":
                    kwargs["method"] = "cg"
                else:
                    # self.loggers["info"].exception()
                    raise e
                run_again = True
                return run_again, kwargs

        if len(w) > 0:
            for w_i in w:
                if w_i.category in [RuntimeWarning, sme.ConvergenceWarning, sme.HessianInversionWarning]:
                    for key in lookup_dict:
                        if key not in kwargs:
                            kwargs[key] = lookup_dict[key]
                            run_again = True
                            return run_again, kwargs
                    if kwargs["maxiter"] < maxiter_limit:
                        run_again = True
                        #kwargs[key] += 50
                        kwargs["maxiter"] = maxiter_limit
                    elif "method" not in kwargs:
                        kwargs["method"] = "bfgs"
                        run_again = True
                    elif kwargs["method"] in methods_list:
                        method_i = methods_list.index(kwargs["method"])
                        if method_i < len(methods_list) - 1:
                            kwargs["method"] = methods_list[method_i + 1]
                            run_again=True
                        else:
                            print("No more solvers!")
                            run_again=False

                    return run_again, kwargs

                return run_again, kwargs
        else:
            return run_again, kwargs

    @classmethod
    def fit_smf_model(cls, smf_model):
        """Runs .fit on smf_model, but checks for linear algebra errors and convergence
        warnings. If there are issues, it changes to a new solver (bfgs instead of
        Newton) and increases the number of iterations.

        Parameters
        ----------
        smf.model : Statsmodels.formula.api model
            Statsmodels model instance (unfitted).

        Returns
        -------
        smf.model.ResultsWrapper
            Fitted model
        """
        run_again = True
        run_counter = 0
        max_runs = 500
        kwargs_dict = {}
        while run_again == True and run_counter < max_runs:
            run_again, kwargs_dict = cls.manage_convergence_warnings(
                smf_model.fit, **kwargs_dict
            )
            run_counter += 1

        return smf_model.fit(disp=0, **kwargs_dict)

    def get_regression_metrics(self, smf_model, mod_key):
        """Calculates regression metrics for given model

        Parameters
        ----------
        smf_model : smf.ModelResults
            smf fit object

        Returns
        -------
        dict
            dictionary of all model results
        """
        model_results = {}
        if "FSS" in self.regression_params:
            if self.regression_params["FSS"] == True:
                model_results["num_features"] = mod_key
        else:
            model_results["model_key"] = mod_key

            model_results["num_features"] = len(
                self.get_x_vals_from_modstring(self.sm_modstrings[mod_key])
            )
        model_results["model"] = self.sm_modstrings[mod_key]

        custom_params = {
            "auc": metrics.roc_auc_score,
            "mnlogit_accuracy": self.mnlogit_accuracy_score_from_y_pred,
            "mnlogit_aucs": self.mnlogit_aucs,
            "mnlogit_mean_auc": self.mnlogit_mean_auc,
            "r2": metrics.r2_score,
        }

        # for info about the model convergence
        mle_settings = {
            "optimizer": "mle_settings",
            "iterations": "mle_retvals",
            "converged": "mle_retvals",
            "fcalls": "mle_retvals",
        }

        y_pred = {"cal": pd.DataFrame(smf_model.predict())}

        if "validation_window" in self.regression_params:
            y_pred["val"] = pd.DataFrame(
                smf_model.predict(exog=self.__model_data__["val"])
            ).reset_index(drop=True)

        for metric in self.regression_params["metrics"]:
            self.loggers["info"].debug(f"       Getting {metric}")
            if metric in custom_params:

                for calval in y_pred:
                    self.loggers["info"].debug(f"       Getting {calval}")

                    y_true = self.__model_data__[calval][
                        self.regression_params["y_col"]
                    ].reset_index(drop=True)
                    model_results[f"{calval}_{metric}"] = custom_params[metric](
                        y_true, y_pred[calval]
                    )
            elif metric in self.SMF_PARAMS_LOOKUP:
                model_results[metric] = getattr(
                    smf_model, self.SMF_PARAMS_LOOKUP[metric]
                )
            else:
                self.loggers["info"].info(f"{metric} unknown. Not calculated.")

        for metric in mle_settings:
            try:
                model_results[metric] = getattr(smf_model, mle_settings[metric])[metric]
            except (KeyError, AttributeError):
                pass

        return model_results

    def run_regression(self, mod_key):
        """Runs regression for given model.

        Parameters
        ----------
        mod_key : int
            Key of given model (also corresponds to number of features in FSS case) to
            run (modstrings are in self.sm_modstrings)
        
        Returns
        -------
        smf.ModelResults
            The ModelResults instance of the model run.
            
        """
        smf_model = getattr(
            smf, self.SMF_FUNCTIONS[self.regression_params["regression_type"]]
        )(self.sm_modstrings[mod_key], data=self.__model_data__["cal"])

        try:
            smf_model = self.fit_smf_model(smf_model)
        except Exception:
            self.loggers["info"].exception()

        return smf_model

    def run_FSS(self):
        """Run Forward Sequential Selection, adding metrics to self.FSS_metrics dict,
        and exporting statsmodels modstrings from selected models

        Returns
        -------
        Dict
            dictionary of model strings from FSS
        """
        self.FSS_metrics = self.forward_sequential_selection(
            self.__model_data__["cal"][self.regression_params["x_cols"]],
            self.__model_data__["cal"][self.regression_params["y_col"]].ravel(),
            name=f"{self.regression_params['name']}",
            scoring_method=self.regression_params["performance_scoring_method"],
            model=self.SKL_FUNCTIONS[self.regression_params["regression_type"]],
        )
        return self.get_modstrings_from_FSS()

    def get_modstrings_from_FSS(self):
        """After running FSS, creates dictionary of statsmodels model strings to run

        Returns
        -------
        Dict
            dictionary of model strings
        """
        features = self.get_features_from_FSS()
        return self.get_sm_modstrings(features, y=self.regression_params["y_col"])

    def get_features_from_FSS(self):
        """Makes dictionary of all features for each model found by FSS.

        Returns
        -------
        Dict
            List of features for each model - key is number of features value is list of
            string names of features.
        """
        feature_names_col = self.FSS_metrics["metric_df"].feature_names
        features = {}
        i = 1
        for feature_tuple in feature_names_col:
            features[i] = list(feature_tuple)
            i += 1
        return features
    
    def get_feature_names_from_FSS(self):
        """Makes list of all features selected via FSS, in order.

        Returns
        -------
        list
            List of features
        """
        feature_list = []

        for i, feat_tuple in enumerate(self.FSS_metrics["metric_df"].feature_names):
            if i==0:
                feature_list.append(feat_tuple[0])
            else:
                new_feature = [x for x in feat_tuple if x not in feature_list]
                feature_list += new_feature
        
        return feature_list


    def get_num_threads_modelled(self):
        """Creates df with number of threads modelled and removed.
        """
        started = False
        num_threads_modelled = {}
        for calval in self.__model_data__:
            num_threads_modelled[calval] = {
                "modelled_threads": len(self.__model_data__[calval])
            }
            if "thresholds" in self.regression_params:
                for colname in self.removed_threads[calval]:
                    num_threads_modelled[calval][f"{colname}_removed_threads"] = len(
                        self.removed_threads[calval][colname]
                    )

        df = pd.DataFrame.from_dict(num_threads_modelled)
        if not started:
            num_threads_modelled_df = df
            started = True
        else:
            num_threads_modelled_df = pd.concat((num_threads_modelled_df, df), axis=1)

        self.num_threads_modelled = num_threads_modelled_df.T

    def output_to_excel(self, outpath, params_to_add=None):
        """Outputs all regression metrics and params to excel spreadsheet.

        Parameters
        ----------
        outpath : str
            path to xlsx outfile
        params_to_add: dict
            Extra params to add to params df (e.g. filenames)
        """
        with pd.ExcelWriter(outpath, engine="xlsxwriter") as writer:
            input_params = self.regression_params.copy()
            if params_to_add is not None:
                for key in params_to_add:
                    input_params[key] = params_to_add[key]
            if self.regression_params["regression_type"] == "mnlogit":
                input_params.pop("performance_scoring_method")
            self.params_dict_to_df(input_params).to_excel(writer, sheet_name="inputs")
            if self.regression_params["regression_type"] == "mnlogit":
                # if mnlogit, output quantile metrics
                self.quantile_metrics.to_excel(writer, sheet_name=f"quantile_metrics")
            if "FSS" in self.regression_params:
                if self.regression_params["FSS"] == True:
                    if type(self.FSS_metrics_df) is dict:
                        self.get_FSS_metrics_df()
                    self.FSS_metrics_df.to_excel(writer, sheet_name="FSS_metrics")

            subreddit_metrics = self.regression_metrics["metrics"]
            if (
                "FSS" not in self.regression_params
                or self.regression_params["FSS"] == False
            ):
                subreddit_metrics.sort_values("num_features", inplace=True)
            min_subset = [
                i for i in subreddit_metrics.columns if ((i == "aic") | (i == "bic"))
            ]
            max_subset = [
                i
                for i in subreddit_metrics.columns
                if (i == "auc") | (i == "validation_auc")
            ]
            (
                subreddit_metrics.style.highlight_min(
                    subset=min_subset, color="green", axis=0
                ).highlight_max(subset=max_subset, color="green", axis=0)
            ).to_excel(writer, sheet_name=f"metrics", index=False)
            if type(self.num_threads_modelled) is dict:
                self.get_num_threads_modelled()
            self.num_threads_modelled.to_excel(writer, sheet_name=f"thread_counts")

            for model in self.regression_metrics["regression_params"]:
                self.regression_metrics["regression_params"][model].to_excel(
                    writer, sheet_name=f"mod{model}"
                )

    def plot_metrics_vs_features(
        self,
        metrics_to_plot,
        ylabel,
        multi_ax=False,
        labels=[],
        name="",
        figsize=(7, 7),
        outfile="",
        xlabel="Number of features",
        show=True,
        title="",
        legend=True,
        legend_loc = (0.6,0.5)
    ):
        """Plot given metrics (aic, auc, bic) on 1 plot.

        Parameters
        ----------
        metrics_to_plot : list(str)
            list of metrics to plot
        y_label : str
            y axis label
        multi_ax : bool, optional
            If multiple metrics, assign True if want separate y axes,
            by default False
        labels: list, optional
            If wish to label metrics, by default empty list
        name : str, optional
            subreddit name, by default ''
        figsize : tuple, optional
            figure size, by default (7,7)
        """
        plt_colours = list(mcolors.TABLEAU_COLORS.keys())
        linestyles = ["solid", "dotted", "dashed"]
        fig, ax = plt.subplots(1, figsize=figsize)

        ax_list = [ax]
        if multi_ax:
            if len(metrics_to_plot) > 1:
                ax_list.append(ax.twinx())
        if not labels:
            labels = metrics_to_plot

        legend_handles = []
        i = 0
        j = 0
        for metric in metrics_to_plot:
            ax_list[i].plot(
                self.regression_metrics["metrics"].index,
                self.regression_metrics["metrics"].loc[:, metric],
                color=plt_colours[i],
                linestyle=linestyles[j],
                label=f"{labels[j]}",
            )
            if multi_ax:
                ax_list[i].set_ylabel(metric)
                i += 1
            else:
                ax_list[i].set_ylabel(ylabel)
            j += 1

        if title=="":
            title = f"{name} {ylabel} vs number of features"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if legend:
            fig.legend(bbox_to_anchor=legend_loc)

        if outfile:
            plt.savefig(outfile)
            plt.close()
        if show:
            plt.show()
        else:
            plt.clf()

    def get_FSS_metrics_df(self):
        """Extracts FSS metrics df from FSS metrics dict.
        Nicer output than dict for writing to csvs.
        """
        df = (
            self.FSS_metrics["metric_df"][
                ["feature_idx", "cv_scores", "avg_score", "feature_names"]
            ]
            .reset_index()
            .rename(columns={"index": "number_features"})
        )
        self.FSS_metrics_df = df

    def plot_FSS(self, figsize=(7, 7)):
        """Plots the forward sequential selection AUC (or other performance measurement)
        vs number of features

        Parameters
        ----------
        figsize : tuple, optional
            size of figure plotted, by default (7,7)
        title : str, optional
            plot title, by default f"{self.regression_params['name']} Sequential Forward Selection"
        """
        fig, ax = plt.subplots(1, figsize=figsize)
        x = self.FSS_metrics["metric_df"].index
        y = self.FSS_metrics["metric_df"].avg_score
        ax.plot(x, y)
        ax.set_title(f"{self.regression_params['name']} Sequential Forward Selection")
        ax.set_xlabel("Number of features")
        ax.set_ylabel(self.regression_params["performance_scoring_method_str"])
        fig.tight_layout()
        plt.show()

    @staticmethod
    def params_dict_to_df(params_dict, to_drop=[]):
        """Tranforms params dict to params df, ideal for excel output

        Parameters
        ----------
        params_dict : dict
            dictionary of input params
        to_drop : list, optional
            list of params to drop (for cleaner output)

        Returns
        -------
        pd.DataFrame
            params df
        """
        # create params df from dict
        params_df = pd.DataFrame.from_dict(params_dict, orient="index").rename(
            columns={0: "input"}
        )
        params_df.index.name = "param"
        if len(to_drop) > 0:
            params_df.drop(labels=to_drop, inplace=True)
        return params_df

    @staticmethod
    def get_x_vals_from_modstring(modstring):
        return [x.strip() for x in modstring.split("~")[1].split("+")]

    @staticmethod
    def get_y_from_modstring(modstring):
        return modstring.split("~")[0].strip()

    @classmethod
    def get_x_vals_from_modstring_dict(cls, modstring_dict):
        mod_list = []
        for modstring in modstring_dict.values():
            mod_list += cls.get_x_vals_from_modstring(modstring)

        return list(set(mod_list))

    @staticmethod
    def get_sm_modstrings(x_list_dict: dict, y: str):
        """With a dictionary of lists of X col names, and string of y col name, makes
        the statsmodels (r-style) string model identifiers

        Parameters
        ----------
        x_list_dict : dict
            Dictionary of lists of x col names
        y : str
            name of y col

        Returns
        -------
        dict
            dictionary of strings of model names
        """
        models = {}

        for feat_num in x_list_dict:
            models[feat_num] = f"{y} ~"
            for i, feat_name in enumerate(x_list_dict[feat_num]):
                if i != 0:
                    models[feat_num] += " +"
                models[feat_num] += f" {feat_name}"
        return models

    @staticmethod
    def separate_float_into_sign_mag(column: pd.Series):
        """separates a column of floats into 2 columns of the sign and magnitude of
        input float

        Parameters
        ----------
        column : pd.Series
            column of floats to separate into signs and magnitudes

        Returns
        -------
        tuple(array, array)
            tuple with first element an array of the signs in the column, second element
            array of magnitudes in the column 
        """
        return (np.sign(column), np.absolute(column))

    @staticmethod
    def forward_sequential_selection(
        X,
        y,
        name="",
        scoring_method="roc_auc",
        model=linear_model.LogisticRegression(),
        cv=None,
    ):
        """Performs forward sequential selection given df of X values to consider
        for features, y column name, an optional name of the data, and scoring method.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of all X features to consider, with 1 col being values of that
            feature.
        y : pd.Series
            Array of successes
        name : str, optional
            Name of given data, by default ''
        scoring_method : str, optional
            Method to use to find best model, by default 'roc_auc'
        model: linear_model class
            Regression model to use (logistic or OLS linear), by default
            linear_model.LogisticRegression()

        Returns
        -------
        Dict
            Dictionary of results of FSS - includes name, the df of metrics, list of
            selected features, and data for plotting
        """

        max_k = len(X.columns)
        k = (1, max_k)

        sfs = SequentialFeatureSelector(
            model, k_features=k, forward=True, scoring=scoring_method, cv=cv
        )

        selected_features = sfs.fit(X, y)
        metric_df = pd.DataFrame.from_dict(
            selected_features.get_metric_dict(), orient="index"
        )

        results = {
            "name": name,
            "metric_df": metric_df,
            "selected_features": selected_features.k_feature_names_,
            "plot_data": selected_features.get_metric_dict(),
        }
        return results

    def regression_params_df(self):
        """Returns df of input params for regressions

        Returns
        -------
        pd.DataFrame
            Table of regression inputs
        """
        df = pd.DataFrame.from_dict(self.regression_params, orient="index").rename(
            columns={0: "input"}
        )
        df.index.name = "param"
        return df

    @staticmethod
    def mnlogit_accuracy_score(estimator, X, y):
        probabilities = pd.DataFrame(estimator.predict_proba(X))
        y_pred = probabilities.idxmax(axis=1)
        value_counts = (y_pred == y).value_counts()
        return value_counts.loc[True] / value_counts.sum()

    @staticmethod
    def mnlogit_accuracy_score_from_y_pred(y_true, y_pred):
        y_pred = y_pred.idxmax(axis=1)
        value_counts = (y_pred == y_true).value_counts()
        return value_counts.loc[True] / value_counts.sum()

    @staticmethod
    def mnlogit_aucs(y_true, y_pred):
        def assign_success_from_quartile(value, quartile_index):
            if value == quartile_index:
                return 1
            else:
                return 0

        auc_vals = []
        for i in y_pred.columns:
            success_data = y_true.apply(assign_success_from_quartile, quartile_index=i)
            success_prediction = y_pred.loc[:, i]
            auc = metrics.roc_auc_score(success_data, success_prediction)
            auc_vals.append(auc)
        return auc_vals

    @classmethod
    def mnlogit_mean_auc(cls, y_true, y_pred):
        return np.mean(cls.mnlogit_aucs(y_true, y_pred))

    @staticmethod
    def get_score(row):
        """
        Combines subject (for posts) and body (for comments) sentiment scores into one
        column

        Parameters
        ----------
        row :
            row in df which has both subject and body sentiment score cols

        Returns
        -------
        pd.Series
            Amalgamated subject and body sentiment scores
        """
        if row.thread_id == row.id:
            return row.subject_sentiment_score
        else:
            return row.body_sentiment_score

    @staticmethod
    def create_param_dict(
        subreddit: str,
        regression_type: str,
        regression_df: pd.DataFrame,
        thread_df: pd.DataFrame,
        **kwargs,
    ):
        """Creates a default params dict to initialise a RedditRegression instance.

        Parameters
        ----------
        subreddit : str
            subreddit name
        regression_type : str
            regression type (linear, logistic or mnlogit)
        regression_df : pd.DataFrame
            regression df for relevant subreddit
        thread_df : pd.DataFrame
            thread df for relevant subreddit

        Returns
        -------
        dict
            params dict
        """
        # fixed regression params
        X_COLS = [
            "sentiment_sign",
            "sentiment_magnitude",
            "time_of_day",
            "weekday",
            "activity_ratio",
            "mean_author_sentiment_sign",
            "mean_author_sentiment_magnitude",
            "author_all_activity_count",
            "mean_author_score",
        ]

        FIXED_PARAMS = {
            "collection_window": 14,
            "model_window": 7,
            "validation_window": 7,
            "FSS": True,
            "x_cols": X_COLS,
            "scale": True,
        }

        # Regression params which depend on type
        QUANTILES = [0.25, 0.5, 0.75]

        THRESHOLDS_SUCCESSFUL = {"author_all_activity_count": 2, "thread_size": 2}
        THRESHOLDS_ALL = {"author_all_activity_count": 2}

        PARAMS_BY_TYPE = {
            "logistic": {
                "regression_type": "logistic",
                "y_col": "success",
                "metrics": ["auc"],
                "thresholds": THRESHOLDS_ALL,
            },
            "linear": {
                "regression_type": "linear",
                "y_col": "log_thread_size",
                "metrics": ["r2"],
                "thresholds": THRESHOLDS_SUCCESSFUL,
            },
            "mnlogit": {
                "regression_type": "mnlogit",
                "y_col": "log_thread_size",
                "metrics": ["mnlogit_accuracy", "mnlogit_aucs", "mnlogit_mean_auc"],
                "thresholds": THRESHOLDS_SUCCESSFUL,
                "quantiles": QUANTILES,
            },
        }

        FIXED_PARAMS.update(PARAMS_BY_TYPE[regression_type])

        regression_params = {
            "name": subreddit,
            "regression_data": regression_df,
            "thread_data": thread_df,
        }
        regression_params.update(kwargs)
        params_to_add = [x for x in FIXED_PARAMS if x not in kwargs]
        for key in params_to_add:
            regression_params[key] = FIXED_PARAMS[key]
        
        # check y col is in reg dfs
        if regression_params['y_col'] not in regression_df:
            y_col = regression_params['y_col']
            if y_col.startswith('log'):
                original_col = y_col.lstrip('log_')
                if original_col in regression_df:
                    # if original column has value higher than 1, can take normal log
                    if regression_df[original_col].min() > 0:
                        regression_df[y_col] = np.log(regression_df[original_col])
                    elif regression_df[original_col].min() == 0:
                        regression_df[y_col] = np.log(regression_df[original_col]+1)
                        print('Taking log(y+1)')
                    else:
                        raise Exception('No log of negative numbers: issue with y column')
                    regression_params['regression_data'] = regression_df
                else:
                    raise Exception('No y column found: nothing to take log of')
            else:
                raise Exception('No y column found')

        return regression_params
