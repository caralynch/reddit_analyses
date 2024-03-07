# data saving imports
import pickle
import os

# time tracking
from datetime import datetime as dt

# multiprocessing
from multiprocessing import Pool

# custom imports
from regression_class import RedditRegression as RR


start = dt.now()
print(f"Start time {start}")

# infiles
# TESTING
TEST_INFILE = "test_data_5_days.p"
test_data = pickle.load(open(TEST_INFILE, 'rb'))
regression_df = test_data["regression_data"]
thread_df = test_data["all_data"]
#REGRESSION_INFILE = "regression_thread_data.p"
#THREAD_INFILE = "clean_5_thread_data.p"

# outfiles
OUTDIR = "regression_outputs"
METRICS_OUTFILE = "regression_metrics"

# subreddits to look at
subreddits = ["books", "crypto", "conspiracy", "politics"]

# regression types to run
regression_types = ["logistic", "linear", "mnlogit"]

# get outdir names for all regression types
out_subdirs = {}
for regtype in regression_types:
    out_subdirs[regtype] = f"{OUTDIR}/{regtype}"

# make out params dict to save spreadsheets
out_params_dict = {}


def run_regression_type(regression_type):
    print(dt.now())
    print(f"\n    ## Regression type: {regression_type}##")
    input_params = regression_params[regression_type].copy()

    # place to store logregs
    subreddit_logregs = {}
    for subreddit in subreddits:
        print(f"#{subreddit}#")
        input_params["name"] = subreddit
        input_params["regression_data"] = regression_df[subreddit]
        input_params["thread_data"] = thread_df[subreddit]

        subreddit_logregs[subreddit] = RR(regression_params=input_params)

        subreddit_logregs[subreddit].main()

    # dump pickle results
    outstring = f"{out_subdirs[regression_type]}/{METRICS_OUTFILE}.p"
    print(dt.now())
    print(f"\n\n\n   DUMPING RESULTS TO \n{outstring}\n\n\n")
    pickle.dump(
        subreddit_logregs, open(outstring, "wb",),
    )
    print(print(dt.now()))


print(dt.now())


# make out directories
for outdirname in out_subdirs:
    if not os.path.isdir(outdirname):
        os.mkdir(outdirname)

# read in files
#print(dt.now())
#print("reading in input files")
# TESTING
#regression_df = pickle.load(open(REGRESSION_INFILE, "rb"))
#thread_df = pickle.load(open(THREAD_INFILE, "rb"))

print(dt.now())
print("Creating parameter dictionaries")
# fixed regression params
X_COLS = [
    "sentiment_sign",
    "sentiment_magnitude",
    "hour",
    "time_in_secs",
    "num_dayofweek",
    "activity_ratio",
    "mean_author_sentiment_sign",
    "mean_author_sentiment_magnitude",
    "author_all_activity_count",
]

fixed_regression_params = {
    "collection_window": 7,
    "model_window": 14,
    "validation_window": 7,
    "FSS": True,
    "x_cols": X_COLS,
    "scale": True,
}

# variable regression params
quantiles = [0.25, 0.5, 0.75]
thresholds2 = {
    "author_all_activity_count": 2,
    "thread_size": 2,
}
thresholds1 = {
    "author_all_activity_count": 2,
}

to_vary = {
    "regression_type": regression_types,
    "y_col": ["success", "thread_size", "thread_size"],
    "metrics": [
        ["auc"],
        ["r2"],
        ["mnlogit_accuracy", "mnlogit_aucs", "mnlogit_mean_auc"],
    ],
    "thresholds": [thresholds1, thresholds2, thresholds2],
    "quantiles": [[], [], quantiles],
}

regression_params = {}
for i, regtype in enumerate(regression_types):
    regression_params[regtype] = fixed_regression_params.copy()
    for key in to_vary:
        regression_params[regtype][key] = to_vary[key][i]

print(dt.now())
print("\nSTARTING REGRESSIONS")
# go through activity thresholds, collection windows and subreddits
with Pool() as pool:
    pool.map(run_regression_type, regression_types)

print(f"Finished: {dt.now()}\nElapsed time {dt.now() - start}")
