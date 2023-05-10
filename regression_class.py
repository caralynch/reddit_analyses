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
from patsy import dmatrices
from itertools import groupby
import os


# for feature selection
from sklearn import linear_model
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

class LogisticRegression(TimestampClass):
    def __init__(self, regression_params:dict):
        """Initialise regression data class

        Parameters
        ----------
        regression_params : dict
            Dictionary with:
            'regression_data': df of regression data,
            'thread_data': df of thread data,
            'collection_window': size of data collection window (used to calc author
                stats), in days,
            'model_window': size of modelling window in days,
            'step': steps between modelling windows in days,
            'FSS': whether to do FSS or not,
            'performance_scoring_method': method used to score FSS,
            'x_cols': features to model,
            'y_col': success column
        """

        self.X_COLS = regression_params['x_cols']

        # if these cols are required they can be calculated
        self.COLUMN_FUNCTIONS = {
            'time_in_secs': super().float_seconds,
            'num_dayofweek': super().get_dayofweek,
            'hour': super().get_hour
        }

        # only need a few cols from regression data
        self.regression_data = regression_params['regression_data'][[
            'thread_id', 'id', 'timestamp', 'author', 'sentiment_score'
            ]]
        
        # want sentiment in thread data to be one col
        self.thread_data = regression_params['thread_data'].apply(self.get_score, axis=1)

        if 'y_col' in regression_params:
            self.y_col = regression_params['y_col']
        else:
            self.y_col = 'success'

        self.time_windows = {}
        for window in ['collection_window', 'model_window', 'step']:
            if window not in regression_params:
                self.time_windows[window] = 7
            else:
                self.time_windows[window] = regression_params[window]
        
        if 'FSS' in regression_params:
            if 'FSS':
                if 'performance_scoring_method' in regression_params:
                    self.perf_scoring_method = regression_params['performance_scoring_method']
                else:
                    self.perf_scoring_method = 'roc_auc'
        
        # get array of dates in dataset
        self.date_array = self.thread_data.timestamp.apply(get_date).unique()

        # create dict for info collection
        self.regression_metrics = {}


    
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
        return hours*60*60 + minutes*60 + seconds
    
    @staticmethod
    def get_dayofweek(timestamp):
        return timestamp.dayofweek
    
    @staticmethod
    def get_hour(timestamp):
        return timestamp.hour