# data saving imports
import pickle
import os

# custom imports
from regression_class import RedditRegression as RR


# infiles
regression_infile = "regression_thread_data.p"
thread_infile = "clean_5_thread_data.p"

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

print('Making regression params dictionaries')
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
            "FSS": True,
        }
        out_params_dict[activity_threshold_size][collection_window_size] = {
            "regression_infile": regression_infile,
            "thread_infile": thread_infile,
        }

# make out directories
for outdirname in [outdir] + list(outdir_names.values()):
    if not os.path.isdir(outdirname):
        os.mkdir(outdirname)

# read in files
print('reading in input files')
regression_df = pickle.load(open(regression_infile, "rb"))
thread_df = pickle.load(open(thread_infile, "rb"))

# place to store logregs
subreddit_logregs = {}

print('\nSTARTING REGRESSIONS')
# go through activity thresholds, collection windows and subreddits
for activity_threshold_size in regression_params_dict:
    print(f"    ###activity_threshold: {activity_threshold_size}###")
    subreddit_logregs[activity_threshold_size] = {}
    for collection_window_size in regression_params_dict[activity_threshold_size]:
        print(f"    ##collection window: {collection_window_size}##")
        subreddit_logregs[activity_threshold_size][collection_window_size] = {}
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

            subreddit_logregs[activity_threshold_size][collection_window_size][
                subreddit
            ] = RR(
                regression_params=regression_params_dict[activity_threshold_size][
                    collection_window_size
                ]
            )

            subreddit_logregs[activity_threshold_size][collection_window_size][
                subreddit
            ].main()


print('SAVING OUTFILES')
# output all files for further analyses in notebook
to_output = {
    "logregs": subreddit_logregs,
    "regression_params": regression_params_dict,
    "out_params": out_params_dict,
    "subdirs": outdir_names,
    "outdir": outdir,
}
pickle.dump(to_output, open(f"{outdir}/logregs_pickle.p", "wb"))
