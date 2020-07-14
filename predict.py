import numpy as np
from propagate import forward_prop


def predict(X, parameters):
    """
    """
    AL, _ = forward_prop(X, parameters)
    return  AL > 0.5
