from skopt.space import Integer, Categorical
import torch
from model.CNN_model import CNN, train
from BO.BO import BaysianOpt, save_results
from data.data_loader import dataloader_
import numpy as np
from visualization import *
import random


def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_random_seeds()

# CNNmodel hyperparameters for optimization
dimensions = [
    Integer(
        8, 48, name="conv_nodes_1"
    ),  # Number of output channels for the first convolutional layer
    Integer(
        8, 48, name="conv_nodes_2"
    ),  # Number of output channels for the second convolutional layer
    Categorical(
        [3, 5], name="kernel_size_1"
    ),  # Kernel size for the first convolutional layer
    Categorical(
        [3, 5], name="kernel_size_2"
    ),  # Kernel size for the second convolutional layer
    Integer(2, 4, name="maxpool_size"),  # Max pooling size
    Categorical(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], name="dropout_rate"
    ),  # Dropout rate
    Integer(32, 512, name="fc_nodes"),  # Number of nodes in the fully connected layer
]

# Optimizer parameters
optimizer_params = {
    "n_calls": 45,
    "n_initial_points": 5,
    "initial_point_generator": "sobol",
    "acq_func": "gp_hedge",
    "n_points": 1000,
    "verbose": True,
}

# Data loader parameters
data_loader_params = {
    "train_size": 500,
    "test_size": 100,
    "val_size": 100,
    "batch_size": 32,
    "shuffle": True,
}

# Number of epochs to train the model
train_epoch = 10

# Load MNIST data
train_loader, val_loader, test_loader = dataloader_(**data_loader_params)

# Perform Bayesian Optimization
OptimizeResult = BaysianOpt(
    CNNmodel=CNN,
    dimensions=dimensions,
    train_epochs=train_epoch,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer_params=optimizer_params,
)

save_results(OptimizeResult, dimensions, optimizer_params)


### PLOTTING ###
# def plot_results(csv_file="BO_results.csv"):
#     # Load results from CSV.
#     df = load_data(csv_file)

#     # Plot accuracy over iterations.
#     plot_metric_over_iterations(
#         df, metric="accuracy", title="Accuracy over Iterations", ylabel="Accuracy"
#     )

#     # Plot acquisition function values over iterations.
#     plot_acquisition_function_values(df)

#     plot_model_size_vs_accuracy(df)


# # Call the plotting function after saving results.
# plot_results()
