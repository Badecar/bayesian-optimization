from skopt import gp_minimize
from skopt.space import Integer, Categorical
import numpy as np
from torch import nn
import model as CNNmodel

def BO(
    Model_class, 
    dimensions, 
    dataloader, 
    val_dataloader, 
    **optimizer_params,
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

    n_calls = optimizer_params['n_calls']
    n_initial_points = optimizer_params['n_initial_points']
    initial_point_generator = optimizer_params['initial_point_generator']
    acquisition = optimizer_params['acquisition']
    n_points = optimizer_params['n_points']
    verbose = optimizer_params['verbose']

    def objective(x):
        model = CNNmodel(**x)

        train_accs, test_accs = model.train_model(
            dataloader, epochs=train_epochs, val_dataloader=test_dataloader
        )
        test_accs = test_accs[-1]
        return - test_accs

    return gp_minimize(
        objective, dimensions,
        n_calls=n_calls, 
        n_initial_points=n_initial_points,
        initial_point_generator=initial_point_generator,
        n_points=n_points,
        verbose=verbose,
    )
