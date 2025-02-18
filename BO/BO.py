from skopt import gp_minimize
from skopt.space import Integer, Categorical
import numpy as np
from torch import nn
import pandas as pd

def BaysianOpt(
    CNNmodel,
    dimensions,
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

    n_calls = optimizer_params["n_calls"]
    n_initial_points = optimizer_params["n_initial_points"]
    initial_point_generator = optimizer_params["initial_point_generator"]
    acquisition = optimizer_params["acquisition"]
    n_points = optimizer_params["n_points"]
    verbose = optimizer_params["verbose"]

    def objective(x):
        model = CNNmodel(**x)

        train_accs, test_accs = model.train_model(
            train_dataloader, epochs=train_epochs, val_dataloader=val_dataloader
        )
        test_accs = test_accs[-1]
        return -test_accs

    return gp_minimize(objective, dimensions, **optimizer_params)

def save_results(optimize_result, dimensions, optimizer_params, filename="BO_results.csv"):

    # Prepare a dictionary to collect results for plotting.
    result_data = {
        "iteration": [],
        "acq_func": [],
        "acq_value": [],
        "accuracy": []
    }

    # Add a column for each hyperparameter.
    for dim in dimensions:
        result_data[dim.name] = []

    # Populate the dictionary with each optimization iteration's data.
    for i, x in enumerate(optimize_result.x_iters):
        result_data["iteration"].append(i + 1)
        result_data["acq_func"].append(optimizer_params["aqcuisition"])
        func_val = optimize_result.func_vals[i]
        result_data["acq_value"].append(func_val)
        # Since the objective returns negative accuracy, recover accuracy.
        result_data["accuracy"].append(-func_val)
        # Save hyperparameter values.
        for j, dim in enumerate(dimensions):
            result_data[dim.name].append(x[j])

    # Create a DataFrame and save to CSV.
    df = pd.DataFrame(result_data)
    df.to_csv(filename, index=False)
    print("Results saved to results.csv")