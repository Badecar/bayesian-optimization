from plots import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

### NOTE: CHANGE PATH ####
path = "visualization/BO_results_CIFAR10.csv"

df = load_data(path)
# df.keys() = hyperparams (ish)

# -----------------
hyperparams = [
    "iteration",
    "acq_func",
    "acq_value",
    "accuracy",
    "conv_nodes_1",
    "conv_nodes_2",
    "kernel_size_1",
    "kernel_size_2",
    "maxpool_size",
    "dropout_rate",
    "fc_nodes",
]

chosen_metrics = [hyperparams[i] for i in (3, 8)]  # NOTE change metrics here

# Plot accuracy over iterations for each acquisition function.
for metric in chosen_metrics:
    plot_metric_over_iterations(
        df,
        metric=metric,
    )
# -----------------

plot_hyperparameter_evolution(df, acq_func="gp_hedge")  # NOTE change acq_func

# EXTRA uncomment to use:
# Scatter plot for model size vs. accuracy.
# plot_maxpool_size_vs_accuracy(df)
