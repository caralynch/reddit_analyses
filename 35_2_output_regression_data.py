import pickle
import pandas as pd
import numpy as np
from regression_class import RedditRegression as RR
import logging
import os
import gc


RESULTS_DIR_PREFIX = "regression_outputs/26_03_2024_"
RESULTS_DIR_SUFFIX = "/results"
OUT_DIR_SUFFIX = "/outputs"
RUN_NAMES = ["c7_m7", "c7_m14", "c14_m7"]
LOGFILE = "regression_outputs/26_03_2024_processing"


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
        param_metrics = result_pickle.regression_params["metrics"]
        metric_cols = result_pickle.regression_metrics["metrics"].columns
        for metric in [x for x in param_metrics if x != "mnlogit_aucs"]:
            logger.info(f"    Plotting {metric}")
            metrics_to_plot = [x for x in metric_cols if metric in x]
            result_pickle.plot_metrics_vs_features(
                metrics_to_plot,
                metric,
                name=f'{filename} {run_type}',
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


if __name__ == "__main__":

    logger = set_up_logger()

    logger.info("Starting")

    for run_name in RUN_NAMES:
        logger.info(f"# {run_name}")
        logger.info("Finding filepaths and making output directories")
        results_dir = RESULTS_DIR_PREFIX + run_name + RESULTS_DIR_SUFFIX
        out_dir = RESULTS_DIR_PREFIX + run_name + OUT_DIR_SUFFIX
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        filepaths = [f"{results_dir}/{x}" for x in os.listdir(results_dir)]
        for filename in filepaths:
            run_outputs(filename, out_dir, logger)
        logger.info(f"# {run_name} completed")

    logger.info(f"FINISHED")
