# data saving imports
import pickle
import os

# custom imports
from regression_class import RedditRegression as RR

# time tracking
from datetime import datetime as dt

# multiprocessing
from multiprocessing import Pool

print(f"Start time {dt.now()}")

# infiles
REGRESSION_INFILE = "regression_thread_data.p"
THREAD_INFILE = "clean_5_thread_data.p"

# outfiles
outdir = "logistic_regression/logregs_23102023"
metrics_outfile = "regression_metrics"

# subreddit to look at
subreddits = ["books", "crypto", "conspiracy", "politics"]

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

# want to run over multiple collection windows and multiple activity thresholds
collection_windows = [1, 3, 7, 14]
activity_thresholds = [0, 1, 2, 5]

# make regression params dict to feed to logreg
regression_params_dict = {}

# store outdir names
outdir_names = {}

# make out params dict to save spreadsheets
out_params_dict = {}

def run_regression_on_collection_windows(collection_window_size):
    print(dt.now())
    print(f"\n    ##collection window: {collection_window_size}##")
    # place to store logregs
    subreddit_logregs = {}
    for subreddit in subreddits:
        print(f"#{subreddit}#")
        regression_params_dict[activity_threshold_size][collection_window_size][
            "name"
        ] = subreddit
        regression_params_dict[activity_threshold_size][collection_window_size][
            "regression_data"
        ] = regression_df[subreddit]
        regression_params_dict[activity_threshold_size][collection_window_size][
            "thread_data"
        ] = thread_df[subreddit]

        subreddit_logregs[subreddit] = RR(
            regression_params=regression_params_dict[activity_threshold_size][
                collection_window_size
            ]
        )

        subreddit_logregs[subreddit].main()

    # dump pickle results
    outstring = (
        f"{outdir_names[activity_threshold_size]}/"
        + f"logregs_a_{activity_threshold_size}_"
        + f"c_{collection_window_size}.p"
    )
    print(dt.now())
    print(f"\n\n\n   DUMPING RESULTS TO \n{outstring}\n\n\n")
    to_output = {
        "logregs": subreddit_logregs,
        "regression_params": regression_params_dict[activity_threshold_size]
        [
            collection_window_size
        ],
        "out_params": out_params_dict[activity_threshold_size][
            collection_window_size
        ],
    }
    pickle.dump(
        to_output, open(outstring, "wb",),
    )
    print(print(dt.now()))
    print("finished dumping\n")


print(dt.now())
print("Making regression params dictionaries")
for activity_threshold_size in activity_thresholds:
    regression_params_dict[activity_threshold_size] = {}
    out_params_dict[activity_threshold_size] = {}
    outdir_names[
        activity_threshold_size
    ] = f"{outdir}/activity_threshold_{activity_threshold_size}"

    for collection_window_size in collection_windows:
        regression_params_dict[activity_threshold_size][collection_window_size] = {
            "regression_type": "logistic",
            "collection_window": collection_window_size,
            "validation_window": 7,
            "FSS": True,
            "performance_scoring_method": "roc_auc",
            "x_cols": X_COLS,
            "y_col": "success",
            "metrics": ["roc_auc", "aic"],
            "activity_threshold": activity_threshold_size,
        }
        out_params_dict[activity_threshold_size][collection_window_size] = {
            "regression_infile": REGRESSION_INFILE,
            "thread_infile": THREAD_INFILE,
        }

# make out directories
for outdirname in [outdir] + list(outdir_names.values()):
    if not os.path.isdir(outdirname):
        os.mkdir(outdirname)

# read in files
print(dt.now())
print("reading in input files")
regression_df = pickle.load(open(REGRESSION_INFILE, "rb"))
thread_df = pickle.load(open(THREAD_INFILE, "rb"))

print(dt.now())
print("\nSTARTING REGRESSIONS")
# go through activity thresholds, collection windows and subreddits
for activity_threshold_size in regression_params_dict:
    print(dt.now())
    print(f"\n    ###activity_threshold: {activity_threshold_size}###")
    with Pool() as pool:
        pool.map(run_regression_on_collection_windows, collection_windows)
        