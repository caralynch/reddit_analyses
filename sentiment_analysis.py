import sys

sys.path.append("/home/cara/Documents/reddit_analysis_code")
from reddit_dataclass import RedditData as reddit
import pickle
import nltk
nltk.download('punkt')

cleaned_datasets = pickle.load(open("cleaned_datasets.p", "rb"))

sentiment_datasets = {}
for key in cleaned_datasets:
    print(f"Analysing {key}")
    sentiment_datasets[key] = cleaned_datasets[key].dataset_sentiment_analysis()

pickle.dump(sentiment_datasets, open("sentiment_datasets.p", "wb"))
