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


class LogisticRegression():
    def __init__(self, regression_params: dict):
        """Initialise regression data class

        Parameters
        ----------
        regression_params : dict
            Dictionary with:
            'name': ideally subreddit name
            'regression_data': df of regression data,
            'thread_data': df of thread data,
            'collection_window': size of data collection window (used to calc author
                stats), in days,
            'model_window': size of modelling window in days,
            'step': steps between modelling windows in days,
            'FSS': whether to do FSS or not,
            'performance_scoring_method': method used to score FSS,
            'x_cols': features to model (used when doing FSS),
            'y_col': success column (used when doing FSS),
            'models': models to use (used when not doing FSS)
            'metrics': list of metrics to output (AUC, AIC,...)
        """

        # if these cols are required they can be calculated
        self.COLUMN_FUNCTIONS = {
            "time_in_secs": TimestampClass.get_float_seconds,
            "num_dayofweek": TimestampClass.get_dayofweek,
            "hour": TimestampClass.get_hour,
        }

        self.regression_data = regression_params["regression_data"]

        # want sentiment in thread data to be one col
        regression_params["thread_data"]["sentiment_score"] = regression_params[
            "thread_data"
        ].apply(self.get_score, axis=1)
        # only need certain cols from thread data
        self.thread_data = regression_params["thread_data"][
            ["thread_id", "id", "timestamp", "author", "sentiment_score"]
        ]

        self.regression_params = {}
        for window in ["collection_window", "model_window", "step"]:
            if window not in regression_params:
                self.regression_params[window] = 7
            else:
                self.regression_params[window] = regression_params[window]

        if "name" in regression_params:
            self.regression_params["name"] = regression_params["name"]
        else:
            self.regression_params["name"] = ""

        if "FSS" in regression_params and (regression_params["FSS"] == True):
            self.regression_params["FSS"] = True
            self.FSS_metrics = {}
            if "performance_scoring_method" in regression_params:
                self.regression_params[
                    "performance_scoring_method"
                ] = regression_params["performance_scoring_method"]
            else:
                self.regression_params["performance_scoring_method"] = "roc_auc"

            # if performing FSS then need x and y cols
            self.regression_params["x_cols"] = regression_params["x_cols"]
            if "y_col" in regression_params:
                self.regression_params["y_col"] = regression_params["y_col"]
            else:
                self.regression_params["y_col"] = "success"
        else:
            # if not performing FSS then need models to run
            self.regression_params["models"] = regression_params["models"]

        if "metrics" in regression_params:
            self.regression_params["metrics"] = regression_params["metrics"]
        else:
            self.regression_params["metrics"] = ["roc_auc"]

        # get array of dates in dataset
        self.date_array = self.thread_data.timestamp.apply(TimestampClass.get_date).unique()

        # create dict for info collection
        self.regression_metrics = {}

    def main(self):
        """
        For each period:
        - gets collection period data and regression model data
        - runs FSS if required and gets sm modstrings
        - iterates through every proposed model:
            - runs logistic regressions
            - calculates metrics required (AIC, BIC, AUC)
        """
        self.period_counter = 1
        # iterate through each time period to model
        for date_index in range(
            0,
            len(self.date_array)
            - (
                self.regression_params["collection_window"]
                + self.regression_params["model_window"]
            ),
            self.regression_params["step"],
        ):
            print(f"# Period {self.period_counter} #")

            # get model data for this period
            self.regression_model_data = self.get_regression_model_data(date_index)

            # if need FSS, run FSS for this time period
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
                logit_out_dict = self.run_logit_regression(mod_key)
                model_results[mod_key] = logit_out_dict["model metrics"]
                param_dict[mod_key] = logit_out_dict["regression params"]

            # convert model metrics dict to df
            model_results = pd.DataFrame.from_dict(model_results, orient="index")

            self.regression_metrics[self.period_counter] = {
                "regression_params": param_dict,
                "metrics": model_results,
            }

            self.period_counter += 1

    def run_logit_regression(self, mod_key):
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
        logit_model = smf.logit(
            self.sm_modstrings[mod_key], data=self.regression_model_data
        ).fit()

        model_results = {}
        model_results["num_features"] = mod_key
        model_results["model"] = self.sm_modstrings[mod_key]

        if "aic" in self.regression_params["metrics"]:
            model_results["aic"] = logit_model.aic
        if "bic" in self.regression_params["metrics"]:
            model_results["bic"] = logit_model.bic
        if "roc_auc" in self.regression_params["metrics"]:
            model_results["auc"] = metrics.roc_auc_score(
                self.regression_model_data.success, logit_model.predict()
            )

        param_df = pd.DataFrame(logit_model.params).rename(columns={0: mod_key})

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
        self.FSS_metrics[self.period_counter] = self.logit_forward_sequential_selection(
            self.regression_model_data[self.regression_params["x_cols"]],
            self.regression_model_data[self.regression_params["y_col"]],
            name=f"{self.regression_params['name']}_period_{self.period_counter}",
            scoring_method=self.regression_params["performance_scoring_method"],
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
        feature_names_col = self.FSS_metrics[self.period_counter][
            "metric_df"
        ].feature_names
        features = {}
        i = 1
        for feature_tuple in feature_names_col:
            features[i] = list(feature_tuple)
            i += 1
        return features

    def plot_metrics_vs_features(self, metrics_to_plot, name="", figsize=(7, 7)):
        """Plot given metrics (aic, roc_auc, bic) on 1 plot.

        Parameters
        ----------
        metrics_to_plot : list(str)
            list of metrics to plot
        name : str, optional
            subreddit name, by default ''
        figsize : tuple, optional
            figure size, by default (7,7)
        """

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
                    self.regression_metrics[period]["metrics"].loc[:, (period, metric)],
                    color=plt_colours[period],
                    linestyle=linestyles[i],
                    label=f"{metric} {period}",
                )
                ax_list[i].set_ylabel(metric)
                ax_list[i].legend()

        ax.set_title(f"{name} information criteria vs number of features all periods")
        ax.set_xlabel("Number of features")

        plt.show()

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

    def get_FSS_metrics_df(self):
        """Extracts FSS metrics df for all time periods from FSS metrics dict.
        Nicer output than dict for writing to csvs.
        """
        started = False
        for period in self.FSS_metrics:
            df = (
                self.FSS_metrics[period]["metric_df"][
                    ["feature_idx", "cv_scores", "avg_score", "feature_names"]
                ]
                .reset_index()
                .rename(columns={"index": "number_features"})
            )
            df.loc[:, "period"] = period
            if not started:
                self.FSS_metrics_df = df
                started = True
            else:
                self.FSS_metrics_df = pd.concat((self.FSS_metrics_df, df))

            # self.FSS_metrics[period].pop('metric_df')
        self.FSS_metrics_df.set_index("period", inplace=True)

    def plot_FSS(
        self, figsize=(7, 7),
    ):
        """Plots the forward sequential selection AUC (or other performance measurement)
        vs number of features for each period considered

        Parameters
        ----------
        figsize : tuple, optional
            size of figure plotted, by default (7,7)
        title : str, optional
            plot title, by default f"{self.regression_params['name']} Sequential Forward Selection"
        """
        fig, ax = plt.subplots(1, figsize=figsize)
        for period in self.FSS_metrics:
            x = self.FSS_metrics[period]["metric_df"].number_features
            y = self.FSS_metrics[period]["metric_df"].avg_score
            ax.plot(x, y, label=period)
        ax.set_title(f"{self.regression_params['name']} Sequential Forward Selection")
        ax.set_xlabel("Number of features")
        ax.set_ylabel(self.regression_params["performance_scoring_method"])
        fig.tight_layout()
        plt.show()

    def get_regression_model_data(self, date_index):
        """Get regression data for model period only, and merge with author data from
        collection period

        Parameters
        ----------
        date_index : int
            Index of start date of collection window
        
        Returns
        -------
        pd.DataFrame
            Regression data that is within model dates, with all required cols
        """

        # find model period dates
        model_dates = self.date_array[
            date_index
            + self.regression_params["collection_window"] : date_index
            + self.regression_params["collection_window"]
            + self.regression_params["model_window"]
        ]
        model_data = self.regression_data[
            self.regression_data.timestamp.apply(self.get_date).isin(model_dates)
        ]

        # get author collection period data
        author_data = self.get_author_collection_data(date_index)

        # combine author collection period data with model data
        model_data = model_data.merge(author_data.reset_index(), on="author")

        # separate mean author sentiment into mag and sign
        col = "mean_author_sentiment"
        (
            model_data[f"{col}_sign"],
            model_data[f"{col}_magnitude"],
        ) = self.separate_float_into_sign_mag(model_data[col])

        # make other required cols
        for col in [
            x for x in self.COLUMN_FUNCTIONS if x in self.regression_params["x_cols"]
        ]:
            model_data[col] = model_data.timestamp.apply(self.COLUMN_FUNCTIONS[col])

        return model_data

    @staticmethod
    def get_x_vals_from_modstring(modstring):
        return [x.strip() for x in modstring.split("~")[1].split("+")]

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

    def get_author_collection_data(self, date_index):
        """Get thread data from collection period only, calculating author activity
        counts and sentiment means

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
            self.thread_data.timestamp.apply(self.get_date).isin(collection_dates)
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

    @staticmethod
    def logit_forward_sequential_selection(X, y, name="", scoring_method="roc_auc"):
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

        Returns
        -------
        Dict
            Dictionary of results of FSS - includes name, the df of metrics, list of
            selected features, and data for plotting
        """

        max_k = len(X.columns)
        k = (1, max_k)

        sfs = SequentialFeatureSelector(
            linear_model.LogisticRegression(),
            k_features=k,
            forward=True,
            scoring=scoring_method,
            cv=None,
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
