from skopt.space import Integer, Categorical, Real
import torch
from model.CNN_model import CNN, train
from BO.BO import BaysianOpt, save_results
from data.data_loader import dataloader_
import numpy as np
import random


def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_random_seeds()

data_set = {"CIFAR10": "(3, 32, 32)", "MNIST": "(1, 28, 28)"}


data_used = "CIFAR10"

# CNNmodel hyperparameters for optimization
dimensions = [
    Categorical(
        [32, 64, 128, 256], name="conv_nodes_1"
    ),  # Number of output channels for the first convolutional layer
    Categorical(
        [32, 64, 128, 256], name="conv_nodes_2"
    ),  # Number of output channels for the second convolutional layer
    Categorical(
        [3, 5], name="kernel_size_1"
    ),  # Kernel size for the first convolutional layer
    Categorical(
        [3, 5], name="kernel_size_2"
    ),  # Kernel size for the second convolutional layer
    Categorical([2, 3, 4], name="maxpool_size"),  # Max pooling size
    Categorical(["max", "mac"], name="pooling_strategy"),  # Pooling strategy
    Categorical([0.2, 0.3, 0.4, 0.5], name="dropout_rate"),  # Dropout rate
    Categorical(
        [128, 256, 512, 1024], name="fc_nodes"
    ),  # Number of nodes in the fully connected layer
    Categorical([data_set[data_used]], name="input_shape"),  # Dataset
    Categorical([1e-4, 1e-3, 1e-2, 1e-1], name="lr"),
]

human_dimensions = [
    Categorical(
        [128], name="conv_nodes_1"
    ),  # Number of output channels for the first convolutional layer
    Categorical(
        [256], name="conv_nodes_2"
    ),  # Number of output channels for the second convolutional layer
    Categorical(
        [5], name="kernel_size_1"
    ),  # Kernel size for the first convolutional layer
    Categorical(
        [5], name="kernel_size_2"
    ),  # Kernel size for the second convolutional layer
    Categorical([3], name="maxpool_size"),  # Max pooling size
    Categorical(["max"], name="pooling_strategy"),  # Pooling strategy
    Categorical([0.3], name="dropout_rate"),  # Dropout rate
    Categorical([512], name="fc_nodes"),  # Number of nodes in the fully connected layer
    Categorical([data_set[data_used]], name="input_shape"),  # Dataset
    Categorical([1e-3], name="lr"),
]


# Optimizer parameters
optimizer_params = {
    "n_calls": 30,
    "n_initial_points": 5,
    "initial_point_generator": "random",  # random, sobol
    "acq_func": "LCB",  # gp_hedge, EI, PI, LCB
    "n_points": 1000,
    "verbose": True,
}

# Data loader parameters
data_loader_params = {
    "dataset": "CIFAR10",
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

# Choose which dimensions to use
human_dimensions_used = True

if human_dimensions_used:
    dimensions = human_dimensions
    file_extra = "_human"
    optimizer_params["n_calls"] = 20
    optimizer_params["n_initial_points"] = 20
    optimizer_params["initial_point_generator"] = "random"
else:
    file_extra = f"_auto_{optimizer_params['acq_func']}"

# Perform Bayesian Optimization
OptimizeResult = BaysianOpt(
    CNNmodel=CNN,
    dimensions=dimensions,
    train_epochs=train_epoch,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer_params=optimizer_params,
)

filename = f"BO_results{data_used}{file_extra}.csv"

save_results(OptimizeResult, dimensions, optimizer_params, filename=filename)
