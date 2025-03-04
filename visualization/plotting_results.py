from plots import*
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### NOTE: CHANGE PATH #### 
path_BO = "/Users/m.brochlips/Programering/AI/Projects/bayesian-optimization/BO/results/BO_results_non_random.csv"
path_RS = "/Users/m.brochlips/Programering/AI/Projects/bayesian-optimization/BO/results/BO_results_random.csv"
df_BO = load_data(path_BO)
df_RS = load_data(path_RS)

iter_lenght = 200 # max 200
mean = 2.03 #human baseline

# -----------------
hyperparams = ['iteration', 'acq_func', 'acq_value', 'accuracy', 'conv_nodes_1', 'conv_nodes_2', 'kernel_size_1', 'kernel_size_2', 'maxpool_size', 'dropout_rate', 'fc_nodes', 'lr']
chosen_metrics = [hyperparams[i] for i in (2,)] #NOTE change metrics here

# Plot accuracy over iterations for each acquisition function.
for metric in chosen_metrics:
    plot_metric_over_iterations2(
        dfbo=df_BO.iloc[24:iter_lenght], 
        dfrs=df_RS.iloc[24:iter_lenght],
        metric=metric,
        style="in_one",
        mean=mean)
# -----------------

# plot_hyperparameter_evolution(df, acq_func="gp_hedge") #NOTE change acq_func

#EXTRA uncomment to use:
# Scatter plot for model size vs. accuracy.
# plot_maxpool_size_vs_accuracy(df)