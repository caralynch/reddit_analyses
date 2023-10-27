# standard imports
import gc

# multiprocessing
from multiprocessing import Pool

# data manipulation imports
import numpy as np
import pandas as pd

# data saving imports
import pickle
import os

# plotting imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib

# custom imports
from regression_class import RedditRegression as RR

# the other 26 file (.py) must be run to run all required regressions

# outfiles
plotting_metrics_out = 'logistic_regression/logregs_26102023/plotting_metrics.p'

# infile parent dir
working_dir = 'logistic_regression/logregs_26102023'


def nice_format_xticks(x_vals_series):
    all_xvals = list(x_vals_series)
    x_vals = []

    for list_xvals in all_xvals:
        for i in list_xvals:
            if i not in x_vals:
                x_vals.append(i)

    separated_out = [x.split('_') for x in x_vals]
    xstrings = []
    for i in separated_out:
        if len(i) == 1:
            xstrings.append(i[0])
        elif len(i) == 2:
            xstrings.append(f'{i[0]}\n{i[1]}')
        else:
            divider = int(len(i)/2)-1
            xstring = ""
            for j in i:
                xstring += j
                if j == i[divider]:
                    xstring += '\n'
                elif j != j[-1]:
                    xstring += ' '
            xstrings.append(xstring)
    return xstrings

def get_regression_params_from_pickles(infile_path):
    collection_window_size = int(infile_path.split('_')[-1].strip('.p'))
    print(f"\n  Collection window: {collection_window_size}")
    print(f"\n  reading in {infile_path}")
    regression_infile = pickle.load(open(infile_path, 'rb'))
    plotting_metrics = {}
    for subreddit in regression_infile['logregs']:
        plotting_metrics[subreddit] = {'collection_window': collection_window_size}
        plotting_metrics[subreddit]['index'] = (
            regression_infile['logregs'][subreddit].regression_metrics[1]["metrics"].index
        )
        plotting_metrics[subreddit]['auc'] = (
            regression_infile['logregs'][subreddit]
            .regression_metrics[1]["metrics"].loc[:, 'auc']
        )
        plotting_metrics[subreddit]['x_names'] = (
            nice_format_xticks(
                regression_infile['logregs'][subreddit]
                .regression_metrics[1]["metrics"].model.apply(
                    regression_infile['logregs'][subreddit].get_x_vals_from_modstring
                )
            )
        )
    del regression_infile
    gc.collect()
    return plotting_metrics

# get infile dirs
infile_dirs = [f"{working_dir}/{x}" for x in os.listdir(working_dir) if os.path.isdir(f"{working_dir}/{x}")]

# iterate through all the infile directories
regressions = {}

for infile_dir in infile_dirs:
    activity_threshold = int(infile_dir[-1])
    print(f"\n\nActivity threshold: {activity_threshold}")
    infiles = [f"{infile_dir}/{x}" for x in os.listdir(infile_dir) if (not os.path.isdir(f"{infile_dir}/{x}")) & (x.startswith('lite'))]
    regressions[activity_threshold] = {}
    print('Starting')
    for infile in infiles:
        collection_window_size = int(infile.split('_')[-1].strip('.p'))
        regressions[activity_threshold][collection_window_size] = get_regression_params_from_pickles(infile)
    
# save plotting params to load to notebook
pickle.dump(regressions, open(plotting_metrics_out, 'wb'))