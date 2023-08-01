# data manipulation imports
import numpy as np
import pandas as pd

# data saving imports
import pickle
import os

# custom imports
from regression_class import RedditRegression as rr

# consider only successful threads
success_only = True

# params
X_COLS = [
    "sentiment_sign",
    "sentiment_magnitude",
    "hour",
    "num_dayofweek",
    "activity_ratio",
    "mean_author_sentiment_sign",
    "mean_author_sentiment_magnitude",
    "log_author_all_activity_count",
]
y_col = "log_thread_size"

# regression params dict
regression_params = {
    "collection_window": 14,
    "validation_window": 7,
    "regression_type": "linear",
    "FSS": True,
    "performance_scoring_method": "r2",
    "x_cols": X_COLS,
    "y_col": y_col,
    "metrics": ["r2"],
    #'activity_threshold': 2,
}


# infiles
regression_infile = "regression_thread_data.p"
thread_infile = "clean_5_thread_data.p"

# outfiles
outdir = "linear_regressions/thread_size/larger_collection_window"
metrics_outfile = "regression_metrics"

# make out params df to save to spreadsheet
out_params = {}
out_params["regression_infile"] = regression_infile
out_params["thread_infile"] = thread_infile

# make out directory
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# read in files
regression_df = pickle.load(open(regression_infile, "rb"))
thread_df = pickle.load(open(thread_infile, "rb"))

# remove thedonald
regression_df.pop("thedonald")

# if success only
if success_only:
    successful_threads = {}
    for subreddit in regression_df:
        successful_threads[subreddit] = regression_df[subreddit][
            regression_df[subreddit].thread_size > 1
        ]
    regression_threads = successful_threads
else:
    regression_threads = regression_df

# place to regression objects
subreddit_regressions = {}

# go through subreddits and run regressions
for subreddit in regression_df:
    print(f"###{subreddit}###")
    regression_params["name"] = subreddit
    regression_params["regression_data"] = regression_threads[subreddit]
    regression_params["thread_data"] = thread_df[subreddit]

    subreddit_regressions[subreddit] = rr(regression_params)
    subreddit_regressions[subreddit].main()

for subreddit in subreddit_regressions:
    for period in subreddit_regressions[subreddit].regression_metrics:
        subreddit_regressions[subreddit].plot_metrics_vs_features_one_period(
            period,
            ["r2"],
            name=f"{subreddit}",
            outfile=f"{outdir}/{subreddit}_p{period}",
            show=False
        )

for subreddit in subreddit_regressions:
    subreddit_outfile = f"{outdir}/{subreddit}_{metrics_outfile}.xlsx"
    subreddit_regressions[subreddit].output_to_excel(
        subreddit_outfile, params_to_add=out_params
    )

pickle.dump(subreddit_regressions, open(f"{outdir}/all_regressions.p", "wb"))
