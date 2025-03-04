import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Seaborn style for nicer plots
sns.set_theme(style="whitegrid")

def load_data(file_path):
    """
    Load results data from a CSV file.
    
    The CSV is expected to have at least the following columns:
      - iteration: the iteration number in the optimization loop.
      - acq_func: name of the acquisition function used.
      - acq_value: the value of the acquisition function at that iteration.
      - accuracy: the validation or test accuracy achieved.
      - loss: the validation or test loss achieved.
      - model_size: a metric representing the model size (e.g., number of parameters).
      
    Additional hyperparameter columns may also be present.
    """
    return pd.read_csv(file_path)



def plot_metric_over_iterations(df, metric):
    """
    Plot the given metric over iterations for each acquisition function.
    
    Parameters:
      - df: DataFrame with columns including 'iteration', 'acq_func', and the metric to plot.
      - metric: The name of the metric column to visualize.
      - title: Title for the plot.
      - ylabel: Label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="iteration", y=metric, hue="acq_func", marker="o")
    plt.title(f'{metric} over iterations')
    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.legend(title="Acquisition Function")
    plt.tight_layout()
    plt.show()



def plot_metric_over_iterations2(dfbo, dfrs, metric, style="in_one", mean=None):
    """
    Plot the given metric over iterations for both datasets with selectable plot style,
    including the option to add a mean line, and highlight the minimum points.

    Parameters:
      - dfbo: First DataFrame (Baysian Optimization).
      - dfrs: Second DataFrame (Random Search).
      - metric: The name of the metric column to visualize.
      - style: 'in_one' to plot both datasets on the same plot, 'on_top' to plot them stacked vertically.
      - mean: A specific value to plot as a horizontal line, or None to disable the mean line.
    """
    # Define dataset names for the legend
    name_bo = 'Bayesian Opt.'
    name_rs = 'Random Search'

    # Calculate the minimum value and corresponding iteration for both datasets
    min_bo_value = dfbo[metric].min()
    min_bo_iteration = dfbo[dfbo[metric] == min_bo_value]["iteration"].values[0]
    
    min_rs_value = dfrs[metric].min()
    min_rs_iteration = dfrs[dfrs[metric] == min_rs_value]["iteration"].values[0]

    if style == "in_one":
        # Plot both datasets on the same plot
        plt.figure(figsize=(10, 6))
        
        # Plot Dataset 1 in blue and Dataset 2 in red (or choose any colors you like)
        sns.scatterplot(data=dfbo, x="iteration", y=metric, label=name_bo, color="blue", marker="o", alpha=0.7)
        sns.scatterplot(data=dfrs, x="iteration", y=metric, label=name_rs, color="red", marker="o", alpha=0.7)

        # Highlight the minimums with larger dots
        plt.scatter(min_bo_iteration, min_bo_value, color="blue", s=200, zorder=5, marker="*",edgecolor="black")
        plt.scatter(min_rs_iteration, min_rs_value, color="red", s=200, zorder=5, marker="*",edgecolor="black",)
        
        # Add mean line if provided
        if mean is not None:
            plt.axhline(mean, color='orange', linestyle='--', label='Human Baseline')
        
        # Title, labels, and grid
        plt.xlabel("Iteration")
        if metric == 'acq_value':
            plt.ylabel("Cross Entropy")
        else:
          plt.ylabel(metric)

        plt.grid(True,alpha=0.7,linestyle="--")

    elif style == "on_top":
        # Plot two separate subplots (one on top of the other)
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        # Plot Dataset 1 on first axis
        sns.lineplot(data=dfbo, x="iteration", y=metric, label=name_bo, color="blue", marker="o", ax=axes[0])
        axes[0].scatter(min_bo_iteration, min_bo_value, color="blue", s=100, label=f'Min of {name_bo} = {min_bo_value:.2f}', zorder=5)
        if mean is not None:
            axes[0].axhline(mean, color='orange', linestyle='--', label=f'Mean = {mean}')
        axes[0].set_title(f'{metric} over iterations - {name_bo}')
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel(metric)
        axes[0].legend(title="Dataset")
        axes[0].grid(True)

        # Plot Dataset 2 on second axis
        sns.lineplot(data=dfrs, x="iteration", y=metric, label=name_rs, color="red", marker="o", ax=axes[1])
        axes[1].scatter(min_rs_iteration, min_rs_value, color="red", s=100, label=f'Min of {name_rs} = {min_rs_value:.2f}', zorder=5)
        if mean is not None:
            axes[1].axhline(mean, color='orange', linestyle='--', label=f'Mean = {mean}')
        axes[1].set_title(f'{metric} over iterations - {name_rs}')
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel(metric)
        axes[1].legend(title="Dataset")
        axes[1].grid(True)

    # Ensure the mean line is included in the legend in the combined plot (if applicable)
    if mean is not None:
        plt.legend(loc="best")

    plt.tight_layout()
    plt.show()

def plot_maxpool_size_vs_accuracy(df):
    """
    Create a scatter plot showing the trade-off between model size and accuracy.
    
    Parameters:
      - df: DataFrame containing at least:
          - 'model_size': Model size metric (e.g., number of parameters).
          - 'accuracy': Accuracy value.
          - 'acq_func': Acquisition function identifier.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="maxpool_size", y="accuracy", hue="acq_func",
                    palette="deep", s=100, alpha=0.8)
    plt.title("Maxpool size vs. Accuracy")
    plt.xlabel("Maxpool size (e.g., number of parameters)")
    plt.ylabel("Accuracy")
    plt.legend(title="Acquisition Function")
    plt.tight_layout()
    plt.show()


def plot_hyperparameter_evolution(df, acq_func):
    """
    Visualize the evolution of grouped hyperparameters over iterations for a single acquisition function.
    
    Parameters:
      - df: DataFrame containing:
            - 'iteration': The iteration number.
            - 'acq_func': Acquisition function identifier.
            - Various hyperparameter columns.
      - acq_func: The acquisition function to filter for plotting.
    """
    df_filtered = df[df['acq_func'] == acq_func]
    
    param_groups = {
        "Kernel & Pooling": ['kernel_size_1', 'kernel_size_2', 'maxpool_size', 'dropout_rate'],
        "Convolutional Nodes": ['conv_nodes_1', 'conv_nodes_2']
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    for (title, params), ax in zip(param_groups.items(), axes):
        for hp in params:
            sns.lineplot(data=df_filtered, x="iteration", y=hp, marker="o", label=hp, ax=ax)
        ax.set_title(f"{title} Evolution for {acq_func}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Hyperparameter Value")
        ax.legend(title="Hyperparameters")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()