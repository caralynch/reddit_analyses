# data manipulation imports
import numpy as np
import pandas as pd

# data saving imports
import pickle
import os
import gc

# plotting imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib

# custom imports
from regression_class import RedditRegression as RR

# outfiles
working_dir = 'logistic_regression/logregs_26102023'
metrics_outfile = "regression_metrics"

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


def plot_metrics_vs_features(
    subreddit_logreg,
    metrics_to_plot,
    name="",
    figsize=(16, 12),
    legend_loc=(0.9, 0.83),
    outfile="",
    xlabel="Features (cumulative)",
    show=True
):
    """Plot given metrics (aic, auc, bic) on 1 plot for specified model period.

    Parameters
    ----------
    period: int
        model period
    metrics_to_plot : list(str)
        list of metrics to plot
    name : str, optional
        subreddit name, by default ''
    figsize : tuple, optional
        figure size, by default (7,7)
    """
    matplotlib.rcParams.update({'font.size': 18})
    plt_colours = list(mcolors.TABLEAU_COLORS.keys())
    fig, ax = plt.subplots(1, figsize=figsize)

    ax_list = [ax]
    if len(metrics_to_plot) > 1:
        ax_list.append(ax.twinx())

    legend_handles = []
    for i, metric in enumerate(metrics_to_plot):
        ax_list[i].plot(
            subreddit_logreg.regression_metrics[1]["metrics"].index,
            subreddit_logreg.regression_metrics[1]["metrics"].loc[:, metric],
            color=plt_colours[i],
            label=f"{metric}",
        )
        ax_list[i].set_ylabel(metric)

    ax.set_title(
        f"{name} information criteria vs features", fontsize=22
    )
    ax.set_xlabel(xlabel)

    # get xtick labels
    x_vals = nice_format_xticks(subreddit_logregs['books'].regression_metrics[1]['metrics'].model.apply(subreddit_logregs['books'].get_x_vals_from_modstring))

    ax.set_xticklabels([0]+x_vals, fontsize=16, rotation=15)

    fig.legend(bbox_to_anchor=legend_loc)

    if outfile:
        plt.savefig(outfile)
    if show:
        plt.show()
    else:
        plt.clf()

# get infile dirs
infile_dirs = [f"{working_dir}/{x}" for x in os.listdir(working_dir) if os.path.isdir(f"{working_dir}/{x}")]

# iterate through all the infile directories
for infile_dir in infile_dirs:
    activity_threshold = int(infile_dir[-1])
    print(f"\n\nActivity threshold: {activity_threshold}")
    infiles = [x for x in os.listdir(infile_dir) if not os.path.isdir(f"{infile_dir}/{x}")]
    for infile in infiles:
        collection_window_size = int(infile.split('_')[-1].strip('.p'))
        print(f"\n  Collection window: {collection_window_size}")
        print(f"\n  reading in {infile}")
        regression_infile = pickle.load(open(f"{infile_dir}/{infile}", 'rb'))

        # output directory
        collection_window_outdir = f"{infile_dir}/collection_window_{collection_window_size}"
        if not os.path.isdir(collection_window_outdir):
            os.mkdir(collection_window_outdir)

        subreddit_logregs = regression_infile['logregs']
        regression_params = regression_infile['regression_params']
        out_params = regression_infile['out_params']

        for subreddit in subreddit_logregs:
            print(f"        {subreddit}")
            print("            plotting")
            plot_metrics_vs_features(
                subreddit_logregs[subreddit], ['auc', 'aic'],
                name=f"{subreddit}", legend_loc=(0.9,0.83), show=False,
                outfile=f"{collection_window_outdir}/{subreddit}_FSS_auc.png"
            )
            subreddit_logregs[subreddit].get_FSS_metrics_df()
            subreddit_outfile = f"{collection_window_outdir}/{subreddit}_{metrics_outfile}.xlsx"

            print(f"         outputting to {subreddit_outfile}")
            subreddit_logregs[subreddit].output_to_excel(subreddit_outfile, params_to_add=out_params)

        # memory management issues
        del regression_infile
        del subreddit_logregs
        del regression_params
        del out_params
        gc.collect()
