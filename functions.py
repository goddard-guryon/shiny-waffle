import numpy as np


def linear(A, W, b):
    """
    Implements the linear function y = W*X + b
    Also returns the original values to store in cache
    """
    return np.dot(W, A) + b, (A, W, b)


def linear_backward(dZ, cache):
    """
    Implements the backward counterpart of the linear function
    """
    A, W, b = cache
    m = A.shape[1]

    dW = np.dot(dZ, A.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA = np.dot(W.T, dZ)

    return dA, dW, db
