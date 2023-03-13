import pandas as pd
import pickle

def get_timediffs(df, func='mean'):
    if len(df) < 2:
        return pd.NaT
    else:
        df = df.sort_values(by='timestamp')
        time_deltas = df['timestamp'].diff()
        if func == 'mean':
            return time_deltas.mean()
        elif func == 'median':
            return time_deltas.median()
        elif func == 'std':
            return time_deltas.std()
        elif func == 'sem':
            return time_deltas.sem()
        else:
            raise KeyError(func)

grouped_timestamps = {}
print("Reading in:")
for key in ['overall', 'comment', 'post']:
    print(f"    {key}")
    grouped_timestamps[key] = pickle.load(open(f"grouped_timestamps_{key}", "rb"))

activation_times = {}
print("Calculating activation times:")
for key in grouped_timestamps:
    timediffs = {}
    print(f"    {key}")
    for op in ['mean', 'sem']:
        timediffs[op] = grouped_timestamps[key].apply(get_timediffs, op)
    activation_times[key] = timediffs

print(f"Dumping")
for key in activation_times:
    print(f"    {key}")
    pickle.dump(activation_times[key], open(f'threshold_activation_times_mean_error_{key}.p', 'wb'))