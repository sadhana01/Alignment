import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib as mpl
import warnings

from os.path import join
from matplotlib import rc
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn as sns

__author__ = 'chetannaik'

# maximum number of parallel process
max_processes = 32

# set of semantic roles in the dataset.
roles = ['undergoer', 'enabler', 'trigger', 'result', 'NONE']
all_roles = ['undergoer', 'enabler', 'trigger', 'result']
positive_roles = ['undergoer', 'enabler', 'trigger', 'result']
labels = ['undergoer', 'enabler', 'trigger', 'result', 'NONE']


# lambda weights to be used in ilp
lambda_1 = 0.9
lambda_2 = 1 - lambda_1

# various paths
project_dir = '/home/slouvan/NetBeansProjects/ILP'
cross_val_dir = join(project_dir, 'data', 'cross-val')

srl_file_name = 'test.srlpredict.json'
srl_file_path = join(project_dir, 'data', srl_file_name)

ilp_out_file_name = 'ilp_predict.json'
ilp_out_path = join(project_dir,'output', ilp_out_file_name)

entailment_data_path = join('entailment_data')
plots_dir = join(project_dir, 'plots')

# plot config (to get LaTeX like plots :)
def set_plot_config():
    rc("grid", alpha=0.9)
    rc("grid", linewidth=0.2)
    rc("grid", linestyle=":")

    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],

        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            ]
        }
    mpl.rcParams.update(pgf_with_latex)

    sns.set_context("paper", font_scale=1.0, rc={'lines.linewidth': 0.75,
                                                 'axes.linewidth': 0.75,
                                                 'text.usetex': True
                                                 })
    sns.set_style("whitegrid", {'font.family': 'serif',
                                'font.serif': ['Palatino']})
