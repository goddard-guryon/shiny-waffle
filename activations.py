import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies sigmoid activation function on given value
    Also returns the original value to store in cache
    """
    return 1/(1+np.exp(-x)), x


def sigmoid_backward(dA: np.ndarray, x:np.ndarray) -> float:
    """
    Applies the derivative of sigmoid activtion for backpropagation
    """
    s = 1 / (1 + np.exp(-x))
    return dA * s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Applies ReLU activation function on given value
    Also returns the original value to store in cache
    """
    return np.maximum(0, x), x


def relu_backward(dA: np.ndarray, x: np.ndarray) -> float:
    """
    Applies the derivtive of ReLU activation for backpropagation
    """
    dZ = np.array(dA, copy=True)
    dZ[x <= 0] = 0
    return dZ


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Applies the softmax activation function on given set
    Also returns the original value to store in cache
    """
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0), x
