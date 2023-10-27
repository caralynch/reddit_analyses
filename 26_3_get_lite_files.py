# standard imports
import gc

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


# infile parent dir
working_dir = 'logistic_regression/logregs_26102023'

# get infile dirs
infile_dirs = [f"{working_dir}/{x}" for x in os.listdir(working_dir) if os.path.isdir(f"{working_dir}/{x}")]

for infile_dir in infile_dirs:
    activity_threshold = int(infile_dir[-1])
    print(f"\n\nActivity threshold: {activity_threshold}")
    infiles = [x for x in os.listdir(infile_dir) if (not os.path.isdir(f"{infile_dir}/{x}")) & (not x.startswith('lite'))]
    for infile in infiles:
        collection_window_size = int(infile.split('_')[-1].strip('.p'))
        print(f"\n  Collection window: {collection_window_size}")
        print(f"\n  reading in {infile}")
        regression_infile = pickle.load(open(f"{infile_dir}/{infile}", 'rb'))

        # delete unneccessary data to save memory and write out "lite" infiles
        for key in ['regression_data', 'thread_data']:
            del regression_infile['regression_params'][key]
        
        pickle.dump(regression_infile, open(f"{infile_dir}/lite_{infile}", 'wb'))
        del regression_infile
        gc.collect()