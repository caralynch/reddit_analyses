import pickle
import pandas as pd
import numpy as np
from regression_class import RedditRegression as RR
import logging
import os
import gc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MAKE_ALL_PLOTS = True
MAKE_COMMUNAL_PLOTS = False

DATE = "09_04_2024"
RESULTS_DIR_PREFIX = f"regression_outputs/{DATE}_"
RESULTS_DIR_SUFFIX = "/results"
OUT_DIR_SUFFIX = "/outputs"
RUN_NAMES = ["c7_m7", "c7_m14", "c14_m7"]
LOGFILE = f"{RESULTS_DIR_PREFIX}_communal_processing"


OUT_DIR_COMBINED = f"regression_outputs/{DATE}_graphs"


def find_metrics_for_plotting(result_pickle):
    param_metrics = result_pickle.regression_params["metrics"]
    metric_cols = result_pickle.regression_metrics["metrics"].columns
    return [x for x in param_metrics if x != "mnlogit_aucs"], metric_cols


def run_outputs(filepath: str, outdir: str, logger):
    logger.info(f"{filepath}")
    filename = os.path.basename(filepath).split(".")[0]
    run_type = " ".join(os.path.dirname(filepath).split("/")[1].split("_")[-2:])
    try:
        logger.info("    Reading in")
        result_pickle = pickle.load(open(filepath, "rb"))
        outpath = f"{outdir}/{filename}_{run_type}.xlsx"
        logger.info(f"    Outputting results to {outpath}")
        result_pickle.output_to_excel(outpath)

        logger.info(f"    Getting metrics")

        metric_list, metric_cols = find_metrics_for_plotting(result_pickle)

        for metric in metric_list:
            logger.info(f"    Plotting {metric}")
            metrics_to_plot = [x for x in metric_cols if metric in x]
            result_pickle.plot_metrics_vs_features(
                metrics_to_plot,
                metric,
                name=f"{filename} {run_type}",
                outfile=f"{outdir}/{filename}_{run_type}_{metric}.png",
                show=False,
            )

        del result_pickle
        gc.collect()
        logger.info(f"    {filepath} finished")
    except:
        logger.exception(f"    Exception occurred with {filepath}")


def set_up_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # stream handler for logging
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # logfile handler
    main_logfile = f"{LOGFILE}.log"
    log_number = 0
    if os.path.isfile(main_logfile):
        make_new = True
        while make_new:
            main_logfile = f"{LOGFILE}_{log_number}.log"
            if os.path.isfile(main_logfile):
                make_new = True
                log_number += 1
            else:
                make_new = False
    f_handler = logging.FileHandler(main_logfile)
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    logger.info(f"Logfile: {main_logfile}")

    return logger


def plot_over_all_runs(
    pickle_dict,
    metrics_to_plot,
    ylabel,
    name="",
    outfile="",
    xlabel="Number of features",
    figsize=(7, 7),
    legend_loc=(0.9, 0.83),
):
    plt_colours = list(mcolors.TABLEAU_COLORS.keys())
    fig, ax = plt.subplots(1, figsize=figsize)
    linestyles = ["solid", "dotted", "dashed"]

    i = 0
    for key in pickle_dict:
        for j, metric in enumerate(metrics_to_plot):
            ax.plot(
                pickle_dict[key].regression_metrics["metrics"].index,
                pickle_dict[key].regression_metrics["metrics"].loc[:, metric],
                color=plt_colours[i],
                linestyle=linestyles[j],
                label=f"{key} {metric}",
            )
        i += 1
    ax.set_title(f"{name} vs number of features")
    ax.set_xlabel(xlabel)
    fig.legend(bbox_to_anchor=legend_loc)

    plt.savefig(outfile)
    plt.close()


def make_common_plots(filepaths, logger):
    logger.info("Graphing combined plots")
    if not os.path.isdir(OUT_DIR_COMBINED):
        os.mkdir(OUT_DIR_COMBINED)
    logger.info("Getting file names")
    filenames = set([os.path.basename(x).split(".")[0] for x in filepaths])
    for x in filenames:
        logger.info(f"    {x}")
        to_graph = [y for y in filepaths if x in y]
        reddit_objs = {}
        for i, filepath in enumerate(to_graph):
            run_name = " ".join(os.path.dirname(filepath).split("/")[1].split("_")[-2:])
            reddit_objs[run_name] = pickle.load(open(filepath, "rb"))
            if i == 0:
                metric_list, metric_cols = find_metrics_for_plotting(
                    reddit_objs[run_name]
                )
        for metric in metric_list:
            logger.info(f"        Plotting {metric}")
            metrics_to_plot = [x for x in metric_cols if metric in x]
            plot_over_all_runs(
                reddit_objs,
                metrics_to_plot,
                metric,
                name=f"{x} {metric}",
                outfile=f"{OUT_DIR_COMBINED}/{x}_{metric}.png",
            )
        logger.info(f"        Done plotting metrics")
    del filenames
    del reddit_objs
    gc.collect()
    logger.info(f"Finished making common plots - all saved to {OUT_DIR_COMBINED}")


if __name__ == "__main__":

    logger = set_up_logger()

    logger.info("Starting")

    all_filepaths = []

    for run_name in RUN_NAMES:
        logger.info(f"# {run_name}")
        logger.info("Finding filepaths and making output directories")
        results_dir = RESULTS_DIR_PREFIX + run_name + RESULTS_DIR_SUFFIX
        out_dir = RESULTS_DIR_PREFIX + run_name + OUT_DIR_SUFFIX
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        filepaths = [f"{results_dir}/{x}" for x in os.listdir(results_dir)]
        all_filepaths += filepaths
        if MAKE_ALL_PLOTS:
            for filename in filepaths:
                run_outputs(filename, out_dir, logger)
        logger.info(f"# {run_name} completed")

    if MAKE_COMMUNAL_PLOTS:
        make_common_plots(all_filepaths, logger)

    logger.info(f"FINISHED")
