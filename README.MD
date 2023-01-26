# Datasets

- Raw data: `/home/cara/Documents/subreddit_data`

- Datasets (nothing removed, raw dataset but thedonald combined months): `datasets.p`

- Datasets with certain authors removed as defined in `authors_to_remove.csv`: `cleaned_datasets.p`

- Same as cleaned_datasets.p but with sentiment cols: `sentiment_datasets.p`

- Preliminary thread data, including unclassified: `thread_data.p`

- Thread data with unclassified threads removed: `clean_thread_data.p`

- Information on thread size, number of participating authors, preliminary levels info: `thread_info.p`

- More in-depth levels info including cumulative sums: `thread_lvls_info.p`

# Analyses

## 1. Clean datasets `clean_datasets.ipynb` - **DONE**
    - check number of removed rows
    - combine any data that needs combining
    - identify most active authors and check whether bots or not
    - remove spam
Outputs: `datasets.p`, `cleaned_datasets.p`
## 2. Perform sentiment analysis `2_sentiment_analysis.py` - **DONE**
Outputs: `sentiment_datasets.p`
## 3. Big-picture, macroscopic analyses `3_whole_subreddit_analyses.ipynb`
### Done so far
    - sentiment score histograms
    - check neutral sentiment activity `neutral_sentiment_activity` dir
    - activity over time
### To do?
    - change timezone in reddit_dataclass graphs over time
    - cumulative score histograms
    - sentiment score over time
    - post activation time?
    - score vs sentiment?


## 4. Microscopic analyses (authors)
check report graphs
### Done
### To do
    - Author activity hists
    - author type (author vs post proportion histogram)
    - author sentiment score
    - author activation time

## 5. Creating threads dataset `5_separate_threads.ipynb` - **DONE**
    - separated data out into threads
    - found unclassified data (missing link in chain)
### Outputs:
    - amendment to `sentiment_datasets.p` (readable parent col)
    - `thread_data.p` (this includes data that has not been assigned a thread level)
    - `clean_thread_data.p`, `unclassified_thread_data.p`
## 6. Thread characteristics `6_thread_characteristics.ipynb` - **DONE**
    - see `thread_analysis.ipnb`
    - thread size by activity histogram
    - thread size by participating users histogram
    - mean and max thread depth
    - thread levels, and level cumulative sums
### Outputs
    - `thread_info.p`
    - `thread_lvls_info.p`
    
### Unsuccessful threads `7_unsuccessful_thread_characteristics.ipynb`
    - see `threads_that_die.ipnb`
    - proportion of unsuccessful threads
    - sentiment distribution of successful vs unsuccessful threads
    - time of day of successful vs unsuccessful thread activity
    - average sentiment over time of s vs uns threads
    - sentiment per chunk of time & day of week

### Authors
    - see `author_stats_tests.ipnb`
    - author probability density vs sentiment score for successful and unsuccessful authors, with different author thresholds
    - 2-sided K-S tests
    - see `thread_authors.ipynb`
    - thread starters vs number of threads started
    - author prob density vs proportion of alive threads
    - comparisons between majority successful and majority unsuccessful authors (prob density vs sentiment score, prob density vs thread count, prob dens vs mean thread size)

### Regression
#### Logistic regression
    - see `logit_with_sm.ipynb`
    




