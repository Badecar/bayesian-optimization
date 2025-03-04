from skopt import gp_minimize
from skopt.space import Integer, Categorical
import numpy as np
from torch import nn
import torch
import torch.optim as optim
from model.CNN_model import train
import pandas as pd
import os


def BaysianOpt(
    CNNmodel,
    dimensions,
    train_epochs,
    train_dataloader,
    val_dataloader,
    optimizer_params,
):
    """
    Perform Bayesian Optimization on a given model class.

    Parameters:
    -----------
    Model_class : class
        The class of the model to be optimized.
    dimensions : list
        List of dimensions for the hyperparameters to be optimized.
    dataloader : DataLoader
        DataLoader for the training data.
    val_dataloader : DataLoader
        DataLoader for the validation data.
    **optimizer_params : dict
        Additional parameters for the optimizer.

    Returns:
    --------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
    """

    # n_calls = optimizer_params["n_calls"]
    # n_initial_points = optimizer_params["n_initial_points"]
    # initial_point_generator = optimizer_params["initial_point_generator"]
    # acquisition = optimizer_params["acquisition"]
    # n_points = optimizer_params["n_points"]
    # verbose = optimizer_params["verbose"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(x):
        # Unpack hyperparameters. Note the new 'lr' parameter is at the end.
        (
            conv_nodes_1,
            conv_nodes_2,
            kernel_size_1,
            kernel_size_2,
            maxpool_size,
            pooling_strategy,
            dropout_rate,
            fc_nodes,
            input_shape,
            lr,
        ) = x

        input_shape = tuple(map(int, input_shape.strip("()").split(",")))

        # Instantiate the model with the given hyperparameters.
        model = CNNmodel(
            conv_nodes_1,
            conv_nodes_2,
            kernel_size_1,
            kernel_size_2,
            maxpool_size,
            pooling_strategy,
            dropout_rate,
            fc_nodes,
            input_shape=(3, 32, 32),  # Use a fixed input shape, or change as needed.
        ).to(device)

        # Use the learning rate from the hyperparameters.
        optimizer = optim.Adam(model.parameters(), lr=lr)

        test_accuracy, test_loss = train(
            model,
            device,
            train_dataloader,
            val_dataloader,
            optimizer,
            train_epochs,
        )

        # Return negative accuracy (if maximizing accuracy).
        return test_loss  # -test_accuracy

    return gp_minimize(objective, dimensions, **optimizer_params)


def save_results(
    optimize_result, dimensions, optimizer_params, filename="BO_results.csv"
):

    # Prepare a dictionary to collect results for plotting.
    result_data = {"iteration": [], "acq_func": [], "acq_value": [], "accuracy": []}

    # Add a column for each hyperparameter.
    for dim in dimensions:
        result_data[dim.name] = []

    # Populate the dictionary with each optimization iteration's data.
    for i, x in enumerate(optimize_result.x_iters):
        result_data["iteration"].append(i + 1)
        result_data["acq_func"].append(optimizer_params["acq_func"])
        func_val = optimize_result.func_vals[i]
        result_data["acq_value"].append(func_val)
        # Since the objective returns negative accuracy, recover accuracy.
        result_data["accuracy"].append(-func_val)
        # Save hyperparameter values.
        for j, dim in enumerate(dimensions):
            result_data[dim.name].append(x[j])

    # Create a DataFrame and save to CSV.
    df = pd.DataFrame(result_data)
    filepath = os.path.join("visualization", filename)
    df.to_csv(filepath, index=False)
    print("Results saved to results.csv")
