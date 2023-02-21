# Datasets

**All files with same name but number 5 inserted are same datasets but with r/politics dataset added**

- Raw data: `/home/cara/Documents/subreddit_data`

- Datasets (nothing removed, raw dataset but thedonald combined months): `datasets.p`

- Datasets with certain authors removed as defined in `authors_to_remove.csv`: `cleaned_datasets.p`

- Same as cleaned_datasets.p but with sentiment cols: `sentiment_datasets.p`

- Preliminary thread data, including unclassified: `thread_data.p`

- Thread data with unclassified threads removed: `clean_thread_data.p`

- Information on thread size, number of participating authors, preliminary levels info: `thread_info.p`

- More in-depth levels info including cumulative sums: `thread_lvls_info.p`

- All, failed and successful posts: `thread_posts.p`

- All, failed and successful posts including EDT timestamp col: `thread_posts_EDT.p`

- Data used for regressions: `regression_thread_data.p`

- regression outputs: `logit_regression_params.xlsx`, `logit_regression_params_v2.xlsx`(includes dataset sizes, skewness, sqrt and without intercept)

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
- time of day of successful vs unsuccessful thread activity + 2-sided K-S test
- average sentiment over time of s vs uns threads
    - hourly
#### To do:
    - sentiment per chunk of time & day of week

### Outputs
Added to `failed_threads` sheet of `dataset_summaries.ods`:
- proportion of failed threads
- statistical tests on the sentiment of successful vs failed posts
- stats on subject sentiment score of failed and successful posts
- hourly activity 2-sided K-S test successful vs failed thread results

- All, failed and successful posts: `thread_posts.p`, `thread_posts_EDT.p`
## Authors
- see `author_stats_tests.ipnb`
- author probability density vs sentiment score for successful and unsuccessful authors, with different author thresholds
- 2-sided K-S tests
- see `thread_authors.ipynb`
- thread starters vs number of threads started
- author prob density vs proportion of alive threads
- comparisons between majority successful and majority unsuccessful authors (prob density vs sentiment score, prob density vs thread count, prob dens vs mean thread size)

## 9. Regressions - `9_regression.ipynb` - **DONE**
### Logistic regression
#### Models
- 'success ~ sentiment_sign',
- 'success ~ sentiment_magnitude',
- 'success ~ sentiment_sign + sentiment_magnitude',
- 'success ~ sentiment_sign*sentiment_magnitude',
- 'success ~ sentiment_sign*sentiment_magnitude + sentiment_sign + sentiment_magnitude',
- 'success ~ log_sentiment_magnitude_plus_one',
- 'success ~ sentiment_sign + log_sentiment_magnitude_plus_one',
- 'success ~ sentiment_sign + sentiment_magnitude + log_sentiment_magnitude_plus_one',
- 'success ~ sentiment_sign*log_sentiment_magnitude_plus_one',
- 'success ~ sentiment_sign*log_sentiment_magnitude_plus_one + sentiment_sign + log_sentiment_magnitude_plus_one'
#### Dataset used
- fullvanilla_models = [
    "success ~ sentiment_sign - 1",
    "success ~ sentiment_sign",
    "success ~ sentiment_magnitude - 1",
    "success ~ sentiment_magnitude",
    "success ~ sentiment_sign + sentiment_magnitude - 1",
    "success ~ sentiment_sign + sentiment_magnitude",
    "success ~ sentiment_sign*sentiment_magnitude - 1",
    "success ~ sentiment_sign*sentiment_magnitude",
    "success ~ sentiment_sign*sentiment_magnitude + sentiment_sign + sentiment_magnitude - 1",
    "success ~ sentiment_sign*sentiment_magnitude + sentiment_sign + sentiment_magnitude",
]

log_models = [
    "success ~ log_sentiment_magnitude_plus_one - 1",
    "success ~ sentiment_sign + log_sentiment_magnitude_plus_one - 1",
    "success ~ sentiment_sign*log_sentiment_magnitude_plus_one - 1",
    "success ~ sentiment_sign*log_sentiment_magnitude_plus_one + log_sentiment_magnitude_plus_one + sentiment_sign - 1",
    "success ~ log_sentiment_magnitude_plus_one",
    "success ~ sentiment_sign + log_sentiment_magnitude_plus_one",
    "success ~ sentiment_sign*log_sentiment_magnitude_plus_one",
    "success ~ sentiment_sign*log_sentiment_magnitude_plus_one + log_sentiment_magnitude_plus_one + sentiment_sign" ,
]

sqrt_models = [
    "success ~ sqrt_sentiment_magnitude - 1",
    "success ~ sentiment_sign + sqrt_sentiment_magnitude - 1",
    "success ~ sentiment_sign*sqrt_sentiment_magnitude - 1",
    "success ~ sentiment_sign*sqrt_sentiment_magnitude + sqrt_sentiment_magnitude + sentiment_sign - 1",
    "success ~ sqrt_sentiment_magnitude",
    "success ~ sentiment_sign + sqrt_sentiment_magnitude",
    "success ~ sentiment_sign*sqrt_sentiment_magnitude",
    "success ~ sentiment_sign*sqrt_sentiment_magnitude + sqrt_sentiment_magnitude + sentiment_sign",
]
- removed neutral sentiment
- with author post activity threshold

### Linear regressions
### Outputs

- `OLS_regression.txt`
- `GLS_regression.txt`
- `regression_thread_data.p`
- `logit_regression_tables.xlsx`

## 10. Further regressions - `10_logistic_regressions.ipynb`
### Models
- vanilla_models = [
    "success ~ sentiment_sign - 1",
    "success ~ sentiment_sign",
    "success ~ sentiment_magnitude - 1",
    "success ~ sentiment_magnitude",
    "success ~ sentiment_sign + sentiment_magnitude - 1",
    "success ~ sentiment_sign + sentiment_magnitude",
    "success ~ sentiment_sign*sentiment_magnitude - 1",
    "success ~ sentiment_sign*sentiment_magnitude",
    "success ~ sentiment_sign*sentiment_magnitude + sentiment_sign + sentiment_magnitude - 1",
    "success ~ sentiment_sign*sentiment_magnitude + sentiment_sign + sentiment_magnitude",
]

- log_models = [
    "success ~ log_sentiment_magnitude_plus_one - 1",
    "success ~ sentiment_sign + log_sentiment_magnitude_plus_one - 1",
    "success ~ sentiment_sign*log_sentiment_magnitude_plus_one - 1",
    "success ~ sentiment_sign*log_sentiment_magnitude_plus_one + log_sentiment_magnitude_plus_one + sentiment_sign - 1",
    "success ~ log_sentiment_magnitude_plus_one",
    "success ~ sentiment_sign + log_sentiment_magnitude_plus_one",
    "success ~ sentiment_sign*log_sentiment_magnitude_plus_one",
    "success ~ sentiment_sign*log_sentiment_magnitude_plus_one + log_sentiment_magnitude_plus_one + sentiment_sign" ,
]

- sqrt_models = [
    "success ~ sqrt_sentiment_magnitude - 1",
    "success ~ sentiment_sign + sqrt_sentiment_magnitude - 1",
    "success ~ sentiment_sign*sqrt_sentiment_magnitude - 1",
    "success ~ sentiment_sign*sqrt_sentiment_magnitude + sqrt_sentiment_magnitude + sentiment_sign - 1",
    "success ~ sqrt_sentiment_magnitude",
    "success ~ sentiment_sign + sqrt_sentiment_magnitude",
    "success ~ sentiment_sign*sqrt_sentiment_magnitude",
    "success ~ sentiment_sign*sqrt_sentiment_magnitude + sqrt_sentiment_magnitude + sentiment_sign",
]
### Outputs
- `logit_regression_params.xlsx`
- `logit_regression_params_v2.xlsx`: includes dataset sizes, skewness, sqrt and without intercept




