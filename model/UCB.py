import numpy as np

def UCB(mean, std, kappa):
    """
    Computes the Upper Confidence Bound (UCB) acquisition function.

    Parameters:
    - mean: Mean of the GP posterior.
    - std_dev: Standard deviation (uncertainty) of the GP posterior.
    - beta: Trade-off parameter (higher = more exploration)

    Returns:
    - Upper confidence bound value.
    """
    return mean + kappa * std