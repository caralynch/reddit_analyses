import pandas as pd
import numpy as np
import powerlaw as pl
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from statistics import mean
import sys

analyzer = SentimentIntensityAnalyzer()

def read_reddit_data(csv_name: str):
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
    for col in ['timestamp', 'thread_id', 'id', 'body', 'subject', 'author', 'image_file', 'domain', 'url', 'image_md5', 'subreddit', 'parent', 'score']:
        data_col_dtypes[col] = str


    data = pd.read_csv(csv_name, usecols=data_col_dtypes.keys())

    # set timestamps to timestamp format
    data_col_dtypes.pop('timestamp')
    data.timestamp=pd.to_datetime(data.timestamp)
    
    # set data types
    for col in data_col_dtypes:
        data[col].astype(data_col_dtypes[col], copy=False)
    
    return data

def compound_body_sentiment(text: str):
    """Calculates sentiment scores of each separate sentence in a text, then averages
    the scores to return the compound sentiment score of the entire text.

    Parameters
    ----------
    text : str
        Text to perform sentiment analysis on.

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
            compound_scores.append(analyzer.polarity_scores(sentence)['compound'])
        
        return mean(compound_scores)
    else:
        return np.nan

def dataset_sentiment_analysis(data: pd.DataFrame):
    """Performs sentiment analysis using VADER of a reddit dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Reddit dataset

    Returns
    -------
    pd.DataFrame
        Reddit dataset with added body and subject sentiment score columns.
    """
    
    # analyse post titles
    data["subject_sentiment_score"] = data.subject.apply(compound_body_sentiment)

    # analyse post and comment bodies
    data["body sentiment score"] = data.body.apply(compound_body_sentiment)

    return data

if __name__ == "__main__":
    print("Reading in data")
    clean_data = read_reddit_data(sys.argv[1])

    print("Analysing sentiment")
    sentiment_data = dataset_sentiment_analysis(clean_data)

    print("Saving to csv")
    sentiment_data.to_csv(f"{sys.argv[1]}_sentiment_data.csv", index=False)
