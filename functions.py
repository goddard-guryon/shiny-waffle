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
    A, W, _ = cache
    m = A.shape[1]

    dW = np.dot(dZ, A.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA = np.dot(W.T, dZ)

    return dA, dW, db


def one_hot_encoded(Y: np.ndarray) -> np.ndarray:
    """
    Apply one-hot encoding on given vector
    """
    database = np.linspace(0, np.max(Y), np.argmax(Y))
    map_to_db = {hot: i for i, hot in enumerate(database)}
    encoding = [map_to_db[num] for num in Y[0]]
    one_hot = []
    for value in encoding:
        vector = [0 for _ in range(len(database))]
        vector[value] = 1
        one_hot.append(vector)
    return np.array(one_hot).T
