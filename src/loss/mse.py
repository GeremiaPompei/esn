import numpy as np

def mse(Y: np.array, P: np.array) -> float:
    """
    Function able to compute the Mean Square Error between predictions and outputs.

    Y: Output values.
    P: Predictions.

    returns:
        float: Value of MSE.
    """
    return ((Y - P) ** 2).mean()