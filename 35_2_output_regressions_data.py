import pickle
import pandas as pd
import numpy as np
from regression_class import RedditRegression as RR
import logging
import os
import sys


# get results dirname
RESULTS_DIR = sys.argv[1]

# get filenames
filenames = os.listdir(RESULTS_DIR)
path_list = [f'{RESULTS_DIR}/{x}' for x in filenames if x.endswith('.p')]

# load pickles
dates = {}
for filename in path_list:
    print(f'## {filename}')
    key = filename.split('/')[-1].removesuffix('.p')
    try:
        print(" Reading in")
        results_pickle = pickle.load(open(f'{filename}', 'rb'))
        print(" Outputting to excel")
        results_pickle.output_to_excel(f"{RESULTS_DIR}/{key}.xlsx")
        print(" Getting dates")
        date_array = results_pickle.date_array
        dates[key] = {'start': date_array.loc[0], 'end': date_array.iloc[-1], 'days': len(date_array)}
    except Exception as e:
        print(f'        Error with {filename}. Passing.')
        pass

print("Saving dates to csv")
pd.DataFrame.from_dict(dates, orient='index').to_csv(f'{RESULTS_DIR}/dataset_dates.csv', 'w')