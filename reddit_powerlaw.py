import enum
from turtle import color
import pandas as pd
import numpy as np
import powerlaw as pl
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from statistics import mean
from matplotlib import pyplot as plt
from matplotlib import dates

class RedditPowerlaw:
    """Class to perform power-law analyses of reddit data.
    """
    def __init__(self, data, significant_fit_param=0.05, name=None, **kwargs) -> None:
        """Initialise a class instance

        Parameters
        ----------
        data : np.array
            1d array of data for power law fitting
        significant_fit_param : float, optional
            Significant fit parameter, by default 0.05
        name : str, optional
            Name of data, by default None
        """
        self.data = data
        self.fit = pl.Fit(data, **kwargs)
        self.significant_fit_param = significant_fit_param
        self.candidate_distributions = []
        self.significance_df = pd.DataFrame()
        self.candidate_params = pd.DataFrame()
        self.name = name

    def find_candidate_distributions(self):
        """Cycle through list of supported distributions to find best fit
        candidates for given dataset.
        """
        distributions = list(self.fit.supported_distributions.keys())
        possible_candidates = list(self.fit.supported_distributions.keys())
        significance_dict = {
            "distribution1": [],
            "distribution2": [],
            "R": [],
            "p": [],
        }
        for distribution1 in distributions:
            distributions.remove(distribution1)
            for distribution2 in distributions:
                (R, p) = self.fit.distribution_compare(distribution1, distribution2)
                significance_dict["distribution1"].append(distribution1)
                significance_dict["distribution2"].append(distribution2)
                significance_dict["R"].append(R)
                significance_dict["p"].append(p)
                if p > self.significant_fit_param:
                    pass
                else:
                    if R < 0:
                        worse = distribution1
                    else:
                        worse = distribution2
                    try:
                        possible_candidates.remove(worse)
                    except ValueError:
                        pass
        self.candidate_distributions = possible_candidates
        self.significance_df = pd.DataFrame.from_dict(significance_dict)

    def get_candidate_fit_params(self, distributions=[]):
        """Gets fit parameters for candidate fit distributions

        Parameters
        ----------
        distributions : list, optional
            List of fits use, otherwise uses best fits, by default []
        """
        if not distributions:
            if not self.candidate_distributions:
                self.find_candidate_distributions()
            distributions = self.candidate_distributions

        distribution_dfs = []
        for distribution in distributions:
            distribution_data = pd.DataFrame.from_dict(
                self.fit.__dict__[distribution].__dict__,
                orient="index",
                columns=[distribution],
            )
            distribution_dfs.append(distribution_data)
        candidate_params = pd.concat(distribution_dfs, axis=1)
        candidate_params.dropna(axis=0, how="all", inplace=True)
        candidate_params.drop(labels="parent_Fit", inplace=True)
        
        """ removing as this removed stuff from the plain power_law fit
        param_rows = [
            x for x in candidate_params.index if (("parameter" in x) & ("name" in x))
        ]
        
        to_remove = []
        for row in param_rows:
            to_remove += list(candidate_params.loc[row, :].values)
        to_remove = list(dict.fromkeys(to_remove))
        if None in to_remove:
            to_remove.remove(None)
        if "lambda" in to_remove:
            to_remove = list(map(lambda x: x.replace("lambda", "Lambda"), to_remove))
        candidate_params.drop(labels=to_remove, inplace=True)
        """
        self.candidate_params = candidate_params

    def plot_fits(self, x_label: str, y_label: str, distributions=[], outfile=None, suptitle=None):
        """Plot data with best fit distributions

        Parameters
        ----------
        x_label : str
            x-axis label
        y_label : str
            y-axis label
        distributions : list, optional
            list of distributions to use, otherwise uses best fit, by
            default []
        outfile : str, optional
            name of file to save plot to, by default None
        suptitle: str, optional
            figure suptitle, by default None
        """
        if not distributions:
            if not self.candidate_distributions:
                self.find_candidate_distributions()
            distributions = self.candidate_distributions
        plots = [x for x in dir(self.fit) if "plot" in x]
        NAMES = {'plot_pdf': 'PDF', 'plot_cdf': 'CDF', 'plot_ccdf': 'CCDF'}
        COLOURS = {
            "original data": "darkred",
            "data": "black",
            "lognormal": "maroon",
            "truncated_power_law": "green",
            "stretched_exponential": "darkorchid",
            "lognormal_positive": "teal",
            'power_law': "darkred",
        }
        fig, axs = plt.subplots(1, 3, figsize=(21, 7))
        handles = []
        labels = []
        for i, plot_function in enumerate(plots):
            getattr(self.fit, plot_function)(
                ax=axs[i], color=COLOURS["data"], label="data"
            )
            for fit_function in distributions:
                getattr(getattr(self.fit, fit_function), plot_function)(
                    ax=axs[i],
                    color=COLOURS[fit_function],
                    linestyle="dashed",
                    label=f"{fit_function}",
                )
            axs[i].set_title(f"{NAMES[plot_function]}")
            axs[i].set_xlabel(f"{x_label}")
            axs[i].set_ylabel(f"{y_label} {plot_function}")
            ax_handles, ax_labels = axs[i].get_legend_handles_labels()
            handles += ax_handles
            labels += ax_labels
        labels_dict = dict(zip(labels, handles))
        fig.legend(labels_dict.values(), labels_dict.keys(), loc="upper right")
        if suptitle:
            fig.suptitle(suptitle)
        elif self.name:
            fig.suptitle(f"{self.name}")
        fig.get_tight_layout()

        if outfile:
            plt.savefig(outfile)

        plt.show()

    def plot_K_S_distance(self, outfile=None):
        """Plots Kolmogorov-Smirnov distance vs x_min for data.

        Parameters
        ----------
        outfile : str, optional
            Name of outfile to save plot to, by default None
        """
        plt.plot(self.fit.xmins, self.fit.Ds, label="all")
        if self.name:
            title_str = f"{self.name} "
        else:
            title_str = ""
        title_str += f"Kolmogorov-Smirnov distance D vs x_min"
        plt.title(title_str)
        plt.xlabel("x_min")
        plt.ylabel("D")
        plt.plot(
            self.fit.xmin,
            self.fit.D,
            marker="+",
            markersize=15,
            color="k",
            label="selected",
        )
        plt.legend()
        if outfile:
            plt.savefig(outfile)
        plt.show()
