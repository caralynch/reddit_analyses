import enum
from turtle import color
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from statistics import mean
from matplotlib import pyplot as plt
from matplotlib import dates
import datetime


class RedditData:
    def __init__(
        self,
        filename=None,
        sentiment_cols=[],
        index_cols=[],
        dataframe=pd.DataFrame(),
        name=None,
        **kwargs,
    ) -> None:
        """_summary_

        Parameters
        ----------
        filename : str, optional
            Name of file with reddit data, by default None
        sentiment_cols : list, optional
            List of names of columns with sentiment data in file, by default empty.
        dataframe : pd.DataFrame, optional
            Dataframe with reddit data if not reading from file, by default pd.DataFrame()

        Raises
        ------
        AttributeError
            If the given dataframe does not have expected columns.
        AttributeError
            If neither a filename nor a dataframe is given.
        """
        self.EXPECTED_COLS = [
            "timestamp",
            "thread_id",
            "id",
            "body",
            "subject",
            "author",
            "image_file",
            "domain",
            "url",
            "image_md5",
            "subreddit",
            "parent",
            "score",
        ]
        if isinstance(filename, str):
            self.data = self.read_reddit_data(
                filename, sentiment_cols=sentiment_cols, index_cols=index_cols, **kwargs
            )
        elif not dataframe.empty:
            if set(self.EXPECTED_COLS).issubset(set(dataframe.columns)):
                self.data = dataframe
            else:
                raise AttributeError(
                    f"Given dataframe does not have expected columns. Columns required are:"
                    f"\n{self.EXPECTED_COLS}"
                )
        else:
            raise AttributeError("No data given.")

        self.data["date"] = self.data.timestamp.dt.date
        self.removed = pd.DataFrame()
        self.comments = self.data[self.data.thread_id != self.data.id]
        self.posts = self.data[self.data.thread_id == self.data.id]
        self.name = name

    def update_comments_posts(self):
        """Updates the comment and post dataframes
        """
        self.comments = self.data[self.data.thread_id != self.data.id]
        self.posts = self.data[self.data.thread_id == self.data.id]

    @property
    def rows(self):
        """Number of rows in data"""
        return len(self.data)

    @property
    def domains(self):
        """Unique domains in data"""
        return self.data.domain.unique()

    @property
    def authors(self):
        """Unique authors in data"""
        return self.data.author.unique()

    @property
    def comment_authors(self):
        """Unique comment authors in data"""
        return self.comments.author.unique()

    @property
    def post_authors(self):
        """Unique post authors in data"""
        return self.posts.author.unique()

    @property
    def times(self):
        """Data start and end timestamps"""
        start = self.data.timestamp.min()
        end = self.data.timestamp.max()
        return (start, end)

    @property
    def comment_bot_rows(self):
        """Comments in data identified as bot submissions"""
        bot_comments = self.data[
            (
                self.data.body.str.find(
                    "I am a bot, and this action was performed automatically"
                )
                != -1
            )
        ]
        return bot_comments

    @property
    def post_bot_rows(self):
        """Posts in data identified as bot submissions"""
        bot_posts = self.data[
            (
                self.data.subject.str.find(
                    "I am a bot, and this action was performed automatically"
                )
                != -1
            )
        ]
        return bot_posts

    def bot_rows(self):
        """Rows in data identified as bot submissions"""
        return pd.concat((self.post_bot_rows, self.comment_bot_rows))

    @property
    def deleted_rows(self):
        """Rows in data that have been removed (by mods) or deleted (by author)"""
        # deleted_posts = self.data[
        # ((self.data.subject == "[deleted]") | (self.data.subject == "[removed]"))
        # ]
        deleted_rest = self.data[
            ((self.data.body == "[deleted]") | (self.data.body == "[removed]"))
        ]
        # return pd.concat((deleted_posts, deleted_rest))
        return deleted_rest

    def summary(self):
        """Generates a df of information about the dataset

        Returns
        -------
        Pandas.DataFrame
            Contains information on the number of rows, comments, posts,
            unique authors, unique domains, bot rows and removed or deleted
            rows as well as the start and end timestamps in the dataset and
            total time elapsed.
        """

        (start_time, end_time) = self.times

        data_dict = {
            "rows": [self.rows],
            "comments": [len(self.comments)],
            "posts": [len(self.posts)],
            "start_time": [start_time],
            "end_time": [end_time],
            "total_time": [end_time - start_time],
            "unique_authors": [len(self.authors)],
            "unique_post_authors": [len(self.post_authors)],
            "unique_comment_authors": [len(self.comment_authors)],
            "unique_domains": [len(self.domains)],
            "bot_posts": [len(self.post_bot_rows)],
            "bot_comments": [len(self.comment_bot_rows)],
            "deleted_comments": [len(self.deleted_rows)],
        }

        return pd.DataFrame.from_dict(data_dict)

    def group_by_col(self, groupby_col: str, cols=None, operation="count"):
        """Groups specified columns of dataframe by given groupby column to get counts,
        mean or other operation.

        Parameters
        ----------
        groupby_col : str
            _description_
        cols : list, optional
            If specified, only uses this subset of the dataframe, by default None
        operation : str, optional
            Specifies operation to perform (count or mean), by default 'count'

        Returns
        -------
        pd.DataFrame
            Grouped dataframe.
        """
        if cols:
            df = self.data[cols]
        else:
            df = self.data
        grouped = df.groupby([groupby_col])
        if operation == "count":
            return grouped.size().reset_index(name="counts")
        elif operation == "mean":
            return grouped.mean().reset_index()
        elif operation == "var":
            return grouped.var().reset_index()
        elif operation == "median":
            return grouped.median().reset_index()
        else:
            raise KeyError(operation)

    def remove_authors(self, filename: str, author_col: str = "author") -> None:
        """Remove specific authors (e.g. bots) from data dataframe, if no removed
        dataframe exists, creates it and adds removed rows.

        Parameters
        ----------
        filename : str
            CSV filename which contains column with str of authors to remove.
        author_col: str
            Name of column with author names.
        """
        authors_to_remove = pd.read_csv(filename, usecols=[author_col])
        if self.removed.empty:
            self.removed = self.data[
                ~self.data.author.isin(authors_to_remove[author_col])
            ]
        else:
            self.removed = pd.concat(
                (
                    self.removed,
                    self.data[~self.data.author.isin(authors_to_remove[author_col])],
                )
            )
        self.data = self.data[~self.data.author.isin(authors_to_remove[author_col])]
        self.update_comments_posts()

    @staticmethod
    def plot_over_time(
        y_data,
        y_label: str,
        time_data,
        ax=None,
        incl_important_dates=True,
        title=None,
        **kwargs,
    ):
        """Makes a plot over time

        Parameters
        ----------
        y_data : np.array
            1d array of y values
        y_label : str
            label for y axis
        time_data : np.array
            1d array of time values
        ax : plt.subplot, optional
            the subplot to graph to, by default None
        incl_important_dates : bool, optional
            whether to include the important dates in the election cycle,
            by default True
        title : str, optional
            Plot title, by default None
        """
        if not ax:
            ax = plt.subplot()

        xaxis = dates.date2num(time_data)
        hfmt = dates.DateFormatter("%d\n%m")
        ax.xaxis.set_major_formatter(hfmt)

        ax.set_xlabel("Date")
        ax.set_ylabel(y_label)
        ax.plot(xaxis, y_data, **kwargs)

        if incl_important_dates:
            important_dates = ["2020-09-29", "2020-10-22", "2020-11-03", "2020-11-07"]
            important_dates = dates.date2num(important_dates)
            date_labels = [
                "first debate",
                "second debate",
                "election day",
                "Biden president-elect",
            ]
            linestyles = ["dotted"] * 4
            for i, val in enumerate(important_dates):
                ax.axvline(
                    x=val, label=date_labels[i], color="k", linestyle=linestyles[i]
                )
                ax.text(
                    val,
                    y_data.max(),
                    date_labels[i],
                    rotation="vertical",
                    fontstyle="italic",
                    ha="right",
                    wrap=True,
                )
        if title:
            ax.set_title(title)

    def plot_hist(
        self,
        col,
        ax=None,
        title=None,
        y_name=None,
        grid=False,
        log=True,
        bins=40,
        **kwargs,
    ):
        """Plots a histogram of given column

        Parameters
        ----------
        col : str
            Column to plot
        ax : plt.axes, optional
            Ax to graph to, by default None
        title : str, optional
            Plot title, by default None
        y_name : str, optional
            Y axis name, by default None
        grid : bool, optional
            Include grid or not, by default False
        log : bool, optional
            Use log scale or not, by default True
        bins : int, optional
            Number of bins, by default 40
        """
        if not ax:
            ax = plt.subplot()
        self.data[col].hist(grid=grid, log=log, bins=bins, ax=ax, **kwargs)
        if title:
            ax.set_title(title)
        if not y_name:
            if self.name:
                y_name = f"{self.name}"
            else:
                y_name = "counts"
        ax.set_xlabel(col)
        ax.set_ylabel(y_name)

    @classmethod
    def dataset_sentiment_analysis(
        cls, data: pd.DataFrame, columns=["body", "subject"]
    ) -> dict:
        """Performs sentiment analysis on a given reddit dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Reddit dataset to perform sentiment analysis on.
        columns: list (optional)
            List of columns to perform sentiment analysis on.

        Returns
        -------
        dict(pd.DataFrame)
            Dictionary with "body" and "subject" elements, with associated sentiment
            scores if applicable.
        """
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = {}

        for column in columns:
            sentiment_scores[column] = data[column].apply(
                cls.compound_body_sentiment, args=(analyzer)
            )

        return sentiment_scores

    def read_reddit_data(
        self,
        csv_name: str,
        sentiment_cols=False,
        index_cols=False,
        shorten_parent=False,
        **kwargs,
    ) -> pd.DataFrame:
        """Read in a reddit dataset csv and set columns to correct formats.

        Parameters
        ----------
        csv_name : str
            Filename of csv to read in.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe of reddit data.
        """

        data_col_dtypes = {
            "score": int,
        }
        for col in self.EXPECTED_COLS:
            data_col_dtypes[col] = str

        if sentiment_cols:
            for col in sentiment_cols:
                data_col_dtypes[col] = float

        if index_cols:
            for col in index_cols:
                data_col_dtypes[col] = float

        data = pd.read_csv(csv_name, usecols=data_col_dtypes.keys(), **kwargs)

        # set timestamps to timestamp format
        data_col_dtypes.pop("timestamp")
        data.timestamp = pd.to_datetime(data.timestamp)

        # set data types
        for col in data_col_dtypes:
            data[col].astype(data_col_dtypes[col], copy=False)

        # change parent ids to true thread ids
        if shorten_parent:
            data["parent"] = data.parent.str[3:]

        return data

    def dataset_sentiment_analysis(self):
        """Performs sentiment analysis using VADER the reddit dataset. Updates
        the dataset to include subject and body sentiment score columns.

        Returns
        -------
        pd.DataFrame
            Reddit dataset with added body and subject sentiment score columns.
        """
        analyzer = SentimentIntensityAnalyzer()

        # analyse post titles
        self.data["subject_sentiment_score"] = self.data.subject.apply(
            self.compound_body_sentiment, args=(analyzer,)
        )

        # analyse post and comment bodies
        self.data["body_sentiment_score"] = self.data.body.apply(
            self.compound_body_sentiment, args=(analyzer,)
        )

        # update comments and posts dataframes
        self.update_comments_posts()

        return self

    @staticmethod
    def compound_body_sentiment(
        text: str, analyzer: SentimentIntensityAnalyzer
    ) -> float:
        """Calculates sentiment scores of each separate sentence in a text, then averages
        the scores to return the compound sentiment score of the entire text.

        Parameters
        ----------
        text : str
            Text to perform sentiment analysis on.
        analyzer: SentimentIntensityAnalyzer
            Instance of VADER sentiment analyzer.

        Returns
        -------
        float
            Compound sentiment score of text.
        np.nan
            NaN object if no string body to analyse.
        """

        # tokenize in case of paragraphs
        if isinstance(text, str):
            sentences = nltk.sent_tokenize(text)

            # get compound score for all sentences
            compound_scores = []
            for sentence in sentences:
                compound_scores.append(analyzer.polarity_scores(sentence)["compound"])

            return mean(compound_scores)
        else:
            return np.nan

    @staticmethod
    def UTC_to_EDT(df_timestamp_col: pd.Series):
        """Conversts timestamps from UTC to EDT

        Parameters
        ----------
        df_timestamp_col : pd.Series
            Column with timestamps in UTC

        Returns
        -------
        pd.Series
            Column with timestamps in EDT
        """

        def get_EDT_time(row):
            return row - pd.Timedelta(4, unit="h")

        return df_timestamp_col.apply(get_EDT_time)

    @staticmethod
    def get_hour(row):
        return row.hour

    @staticmethod
    def get_time(row):
        return row.time()

    @staticmethod
    def get_dayofweek(df_timestamp_col: pd.Series):
        def get_day(row):
            return row.dayofweek

        daysofweek = {0: "M", 1: "Tu", 2: "W", 3: "Th", 4: "F", 5: "Sa", 6: "Su"}

        return df_timestamp_col.apply(get_day).map(daysofweek)

    @staticmethod
    def conv_to_datetime(row):
        my_day = datetime.date(2020, 1, 1)
        return datetime.datetime.combine(my_day, row)

    @staticmethod
    def conv_date2num(row):
        return row.date2num(time_data)
