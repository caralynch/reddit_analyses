import numpy as np
import pandas as pd
from reddit_dataclass import RedditData as reddit
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as scpstat
import matplotlib.dates as dates
import datetime
from sklearn import metrics
import statsmodels.formula.api as smf
import statsmodels.api as sm
from itertools import groupby


# for feature selection
from sklearn import linear_model
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

"""
### TODO ###

Add multinomial logistic regression:
    - assume threads will have already been binned, and the bins are numbered (indices)
    - add function to calculate AUC for multinomial logreg
        - need to assign success for each quartile
        - also get success prediction for each quartile (mnlogit predict yields 4 numbers - one for each bin, but check this)
        - get AUCs for all quartiles (from sklearn metrics)
        - use average over all classes??
Add domain tracker - e.g. domain frequency or something?
"""
class RedditRegression:
    def __init__(self, regression_params: dict):
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
            'models': models to use (used when not doing FSS)
            'metrics': list of metrics to output (AUC, AIC,...)
            'activity_threshold': nb of activities per week per author to include
                                (optional)
        """

        # if these cols are required they can be calculated
        self.COLUMN_FUNCTIONS = {
            "time_in_secs": TimestampClass.get_float_seconds,
            "num_dayofweek": TimestampClass.get_dayofweek,
            "hour": TimestampClass.get_hour,
        }

        self.SMF_FUNCTIONS = {"logistic": "logit", "linear": "ols", "mnlogit": "mnlogit"}

        self.SKL_FUNCTIONS = {
            "logistic": linear_model.LogisticRegression(),
            "linear": linear_model.LinearRegression(),
            "mnlogit": linear_model.LinearRegression(),
        }

        self.regression_params = {}

        self.regression_data = regression_params["regression_data"]

        # want sentiment in thread data to be one col
        regression_params["thread_data"]["sentiment_score"] = regression_params[
            "thread_data"
        ].apply(self.get_score, axis=1)
        # only need certain cols from thread data
        self.thread_data = regression_params["thread_data"][
            ["thread_id", "id", "timestamp", "author", "sentiment_score"]
        ]

        # if doing validation, add validation window
        if "validation_window" in regression_params:
            self.regression_params["validation_window"] = regression_params["validation_window"]

        if "name" in regression_params:
            self.regression_params["name"] = regression_params["name"]
        else:
            self.regression_params["name"] = ""

        if "regression_type" not in regression_params:
            regression_params["regression_type"] = "logistic"

        if "FSS" in regression_params and (regression_params["FSS"] == True):
            self.regression_params["FSS"] = True
            self.FSS_metrics = {}
            self.FSS_metrics_df = {}
            if "performance_scoring_method" in regression_params:
                self.regression_params[
                    "performance_scoring_method"
                ] = regression_params["performance_scoring_method"]
            else:
                if regression_params["regression_type"] == "logistic":
                    self.regression_params["performance_scoring_method"] = "roc_auc"
                else:
                    self.regression_params["performance_scoring_method"] = "r2"

            # if performing FSS then need x and y cols
            self.regression_params["x_cols"] = regression_params["x_cols"]
            if "y_col" in regression_params:
                self.regression_params["y_col"] = regression_params["y_col"]
            else:
                self.regression_params["y_col"] = "success"
        else:
            # if not performing FSS then need models to run
            self.regression_params["models"] = regression_params["models"]
            # if input model strings, then find the x cols
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

        if "metrics" in regression_params:
            self.regression_params["metrics"] = regression_params["metrics"]
        else:
            self.regression_params["metrics"] = ["roc_auc"]

        # model type assigns model function to use
        self.regression_params["regression_type"] = regression_params["regression_type"]

        # get array of dates in dataset
        date_array = self.thread_data.timestamp.apply(TimestampClass.get_date).unique()

        # the dataset may be missing days - so create df from start and end dates
        date_array = pd.date_range(start=date_array[0], end=date_array[-1])
        self.date_array = pd.DataFrame(date_array)[0].apply(TimestampClass.get_date)

        for window in ["collection_window", "model_window"]:
            if window not in regression_params:
                if window == "collection_window":
                    self.regression_params[window] = 7
                elif "validation_window" in self.regression_params:
                    self.regression_params[window] = len(self.date_array) - (
                        self.regression_params["validation_window"]
                        + self.regression_params["collection_window"]
                    )
                else:
                    self.regression_params[window] = 7
            else:
                self.regression_params[window] = regression_params[window]

        # author activity threshold
        if "activity_threshold" in regression_params:
            self.regression_params["activity_threshold"] = regression_params[
                "activity_threshold"
            ]
            self.removed_threads = {}

        # create dict for info collection
        self.regression_metrics = {}
        self.model_data = {}
        self.num_threads_modelled = {}

    def calc_author_thread_counts(self):
        """Update thread data df with author activity, post and comment counts. These
        are calculated by looking at the 7 days preceding the day of current activity.

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
        ]

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
            collection_thread_data = collection_thread_data[
                collection_thread_data.author.isin(day_thread_data.author)
            ]

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
            author_mean_sentiment = (
                thread_activity["all_activity"][["author", "sentiment_score"]]
                .groupby("author")
                .mean()
                .rename(columns={"sentiment_score": f"mean_author_sentiment"})
            )

            # combine to form author info df
            author_info = pd.concat((author_activity, author_mean_sentiment,), axis=1,)

            # convert to dict of mapping dicts to add to thread data from day
            author_info_maps = author_info.to_dict()

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
        col = "log_author_all_activity_count"
        if col in self.regression_params["x_cols"]:
            self.thread_data["log_author_all_activity_count"] = np.log(
                self.thread_data.loc[:, "author_all_activity_count"] + 1
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

    def main(self):
        """
        Calculates author count data from thread data with a rolling collection window.
        - gets regression model data
        - runs FSS if required and gets sm modstrings
        - iterates through every proposed model:
            - runs logistic regressions
            - calculates metrics required (AIC, BIC, AUC)
        """
        # get thread author data
        self.calc_author_thread_counts()

        date_index = 0

        # get model data
        self.get_regression_model_data(date_index)

        # get validation data
        if "validation_window" in self.regression_params:
            self.get_regression_model_data(date_index, calval="val")

        # if need FSS, run FSS
        if "FSS" in self.regression_params:
            print(f"Running FSS")
            self.sm_modstrings = self.run_FSS()
        else:
            self.sm_modstrings = self.regression_params["models"]
            # TODO fill in this section

        # iterate through each model
        model_results = {}
        param_dict = {}
        for mod_key in self.sm_modstrings:
            print(f"Model {mod_key}")
            regression_out_dict = self.run_regression(mod_key)
            model_results[mod_key] = regression_out_dict["model_metrics"]
            param_dict[mod_key] = regression_out_dict["regression_params"]

        # convert model metrics dict to df
        model_results = pd.DataFrame.from_dict(model_results, orient="index")

        self.regression_metrics = {
            "regression_params": param_dict,
            "metrics": model_results,
        }


    def get_regression_model_data(self, date_index, calval="cal"):
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

        # need to add author_all_activity_count for filtering purposes if thresholding
        if ("activity_threshold" in self.regression_params) & (
            "author_all_activity_count" not in thread_data_cols
        ):
            thread_data_cols += ["author_all_activity_count"]

        # only if there are actually thread cols to add (so if considering author
        # features)
        if len(thread_data_cols) > 1:
            model_data = self.merge_author_and_model_data(
                model_data, thread_data_cols, calval
            )

        # make other required cols
        for col in [
            x for x in self.COLUMN_FUNCTIONS if x in self.regression_params["x_cols"]
        ]:
            model_data[col] = model_data.timestamp.apply(self.COLUMN_FUNCTIONS[col])

        self.model_data[calval] = model_data

        if calval == "cal":
            self.regression_model_data = model_data
        else:
            self.validation_data = model_data

    def merge_author_and_model_data(
        self, model_data: pd.DataFrame, thread_data_cols: list, calval: str
    ):
        """Merges author data calculated with thread_data, with the given model_data.
        Performs a left hand merge with model data, such that only threads are left.
        Only merges columns that are specified in the regression params.
        Also performs author activity thresholding.

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
            model data with required author columns and thresholds applied
        """
        model_data = model_data.merge(
            self.thread_data[thread_data_cols],
            left_on="thread_id",
            right_on="id",
            how="left",
        )

        # if thresholding by author activity, remove threads where author activity
        # is less than threshold as these cannot be modelled
        if "activity_threshold" in self.regression_params:
            threshold = self.regression_params["activity_threshold"]
            #print(f"Performing thresholding")
            self.removed_threads[calval] = model_data.loc[
                model_data.author_all_activity_count < threshold, :
            ]
            model_data = model_data.loc[
                model_data.author_all_activity_count >= threshold, :
            ]

        # drop cols unnecessary for modelling
        to_drop = ["id"]
        if ("author_all_activity_count" in thread_data_cols) & (
            "author_all_activity_count" not in self.regression_params["x_cols"]
        ):
            to_drop += ["author_all_activity_count"]
        model_data.drop(labels=to_drop, axis=1, inplace=True)

        return model_data

    def run_regression(self, mod_key):
        """Runs logistic regression for given model and updates model result and
        parameters dicts with metrics and regression parameters for given model.

        Parameters
        ----------
        mod_key : int
            Key of given model (also corresponds to number of features in FSS case) to
            run (modstrings are in self.sm_modstrings)
        
        Returns
        -------
        Dict
            With 'model metrics' and 'regression params' keys
        """
        smf_model = (
            getattr(smf, self.SMF_FUNCTIONS[self.regression_params["regression_type"]])(
                self.sm_modstrings[mod_key], data=self.regression_model_data
            )
        ).fit()

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

        SMF_PARAMS_LOOKUP = {
            "aic": "aic",
            "bic": "bic",
            "r2": "rsquared",
        }
        for metric in self.regression_params["metrics"]:
            if metric == "roc_auc":
                model_results["auc"] = metrics.roc_auc_score(
                    self.regression_model_data.success, smf_model.predict()
                )
            else:
                model_results[metric] = getattr(smf_model, SMF_PARAMS_LOOKUP[metric])

        stderr = pd.Series(
            np.sqrt(np.diag(smf_model.cov_params())), index=smf_model.params.index
        )
        param_df = pd.DataFrame([smf_model.params, stderr, smf_model.pvalues]).T.rename(
            columns={0: "param", 1: "stderr", 2: "pvalue"}
        )

        # get test dataset results
        if "validation_window" in self.regression_params:
            y_test_prediction = smf_model.predict(exog=self.validation_data)
            if "roc_auc" in self.regression_params["metrics"]:
                model_results["validation_auc"] = metrics.roc_auc_score(
                    self.validation_data.success, y_test_prediction
                )
            else:
                model_results["validation_r2"] = metrics.r2_score(
                    self.validation_data.success, y_test_prediction
                )

        out_dict = {"model_metrics": model_results, "regression_params": param_df}
        return out_dict

    def run_FSS(self):
        """Run Forward Sequential Selection, adding metrics to self.FSS_metrics dict,
        and exporting statsmodels modstrings from selected models

        Returns
        -------
        Dict
            dictionary of model strings from FSS
        """
        self.FSS_metrics = self.forward_sequential_selection(
            self.regression_model_data[self.regression_params["x_cols"]],
            self.regression_model_data[self.regression_params["y_col"]],
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
        feature_names_col = self.FSS_metrics[
            "metric_df"
        ].feature_names
        features = {}
        i = 1
        for feature_tuple in feature_names_col:
            features[i] = list(feature_tuple)
            i += 1
        return features

    def get_num_threads_modelled(self):
        """Creates df with number of threads modelled and removed.
        """
        started = False
        num_threads_modelled = {}
        for calval in self.model_data:
            num_threads_modelled[calval] = {
                "modelled_threads": len(self.model_data[calval])
            }
            if "activity_threshold" in self.regression_params:
                num_threads_modelled[calval]["removed_threads"] = len(
                    self.removed_threads[calval]
                )
        df = pd.DataFrame.from_dict(num_threads_modelled)
        if not started:
            num_threads_modelled_df = df
            started = True
        else:
            num_threads_modelled_df = pd.concat(
                (num_threads_modelled_df, df), axis=1
            )

        self.num_threads_modelled = num_threads_modelled_df.T

    def output_to_excel(self, outpath, params_to_add={}):
        """Outputs all regression metrics and params to excel spreadsheet.

        Parameters
        ----------
        outpath : str
            path to xlsx outfile
        params_to_add: dict
            Extra params to add to params df (e.g. filenames)
        """
        with pd.ExcelWriter(outpath, engine="xlsxwriter") as writer:
            for key in params_to_add:
                self.regression_params[key] = params_to_add[key]
            self.params_dict_to_df(self.regression_params).to_excel(
                writer, sheet_name="inputs"
            )
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
                i
                for i in subreddit_metrics.columns
                if ((i == "aic") | (i == "bic"))
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
                self.regression_metrics["regression_params"][
                    model
                ].to_excel(writer, sheet_name=f"mod{model}")

    def plot_metrics_vs_features(
        self,
        metrics_to_plot,
        ylabel,
        multi_ax = False,
        labels = [],
        name="",
        figsize=(7, 7),
        legend_loc=(0.9, 0.83),
        outfile="",
        xlabel="Number of features",
        show=True
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
                color=plt_colours[j],
                label=f"{labels[j]}",
            )
            if multi_ax:
                ax_list[i].set_ylabel(metric)
                i += 1
            else:
                ax_list[i].set_ylabel(ylabel)
            j += 1
                

        ax.set_title(
            f"{name} information criteria vs number of features"
        )
        ax.set_xlabel(xlabel)
        fig.legend(bbox_to_anchor=legend_loc)

        if outfile:
            plt.savefig(outfile)
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


    def plot_FSS(
        self, figsize=(7, 7),
    ):
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
        ax.set_ylabel(self.regression_params["performance_scoring_method"])
        fig.tight_layout()
        plt.show()

    @staticmethod
    def params_dict_to_df(params_dict):
        """Tranforms params dict to params df, ideal for excel output

        Parameters
        ----------
        params_dict : dict
            dictionary of input params

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
        X, y, name="", scoring_method="roc_auc", model=linear_model.LogisticRegression()
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
            model, k_features=k, forward=True, scoring=scoring_method, cv=None,
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

    def get_author_collection_data(self, date_index):
        """DEPRECATED
        Get thread data from collection period only, calculating author activity
        counts and sentiment means. This should only be used when collection period and
        model period are completely separate - e.g. no rolling time windows.

        Parameters
        ----------
        date_index : int
            Index of start date of collection window

        Returns
        -------
        pd.DataFrame
            Author activity count, activity ratio and mean sentiment score throughout
            collection period
        """
        # get collection dates
        collection_dates = self.date_array[
            date_index : date_index + self.regression_params["collection_window"]
        ]

        # get thread data in collection dates
        thread_collection_data = self.thread_data[
            self.thread_data.timestamp.apply(TimestampClass.get_date).isin(
                collection_dates
            )
        ]

        # separate by activity
        thread_activity = {
            "all_activity": thread_collection_data,
            "post": thread_collection_data[
                thread_collection_data.thread_id == thread_collection_data.id
            ],
            "comment": thread_collection_data[
                thread_collection_data.thread_id != thread_collection_data.id
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
        author_mean_sentiment = (
            thread_activity["all_activity"][["author", "sentiment_score"]]
            .groupby("author")
            .mean()
            .rename(columns={"sentiment_score": f"mean_author_sentiment"})
        )

        # combine to form author info df
        author_info = pd.concat(
            (
                author_activity[["author_all_activity_count", "activity_ratio"]],
                author_mean_sentiment,
            ),
            axis=1,
        )

        return author_info


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


# ARCHIVE
"""
    def plot_metrics_vs_features_all_periods(
        self, metrics_to_plot, name="", figsize=(7, 7)
    ):
        '''Plot given metrics (aic, auc, bic) on 1 plot over all time periods.
        Parameters
        ----------
        metrics_to_plot : list(str)
            list of metrics to plot
        name : str, optional
            subreddit name, by default ''
        figsize : tuple, optional
            figure size, by default (7,7)
        '''

        linestyles = ["solid", "dotted", "dashed"]
        plt_colours = list(mcolors.TABLEAU_COLORS.keys())
        fig, ax = plt.subplots(1, figsize=figsize)
        if len(metrics_to_plot) > 1:
            ax_list = [ax, ax.twinx()]
        else:
            ax_list = [ax]

        for period in self.regression_metrics:
            for i, metric in enumerate(metrics_to_plot):
                ax_list[i].plot(
                    self.regression_metrics[period]["metrics"].index,
                    self.regression_metrics[period]["metrics"].loc[:, metric],
                    color=plt_colours[period],
                    linestyle=linestyles[i],
                    label=f"{metric} {period}",
                )
                ax_list[i].set_ylabel(metric)
                ax_list[i].legend()

        ax.set_title(f"{name} information criteria vs number of features all periods")
        ax.set_xlabel("Number of features")

        plt.show()

"""