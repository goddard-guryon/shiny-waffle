import numpy as np
from typing import Union, Optional
from activations import sigmoid, relu, sigmoid_backward, relu_backward
from functions import linear, linear_backward


def linear_act_forward(A, W, b, act):
    """
    Implements the linear and activation functions of a single node
    Also, returns the original values for storage in cache
    """
    if act == "sigmoid":
        Z, lin_cache = linear(A, W, b)
        A, act_cache = sigmoid(Z)
    elif act == "relu":
        Z, lin_cache = linear(A, W, b)
        A, act_cache = relu(Z)

    cache = (lin_cache, act_cache)

    return A, cache


def linear_act_forward_with_dropout(A, W, b, act, kappa):
    """
    Implements the linear and activation functions with dropout regularization
    Also, returns the original values for storage in cache
    """
    if act == "sigmoid":
        Z, lin_cache = linear(A, W, b)
        A, act_cache = sigmoid(Z)
    elif act == "relu":
        Z, lin_cache = linear(A, W, b)
        A, act_cache = relu(Z)

    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < kappa).astype(int)
    A = (A * D) / kappa

    cache = (lin_cache, act_cache)

    return A, cache, D


def linear_act_backward(dA, cache, act, kappa):
    """
    Implements the linear and activation function derivatives of single node
    """
    lin_cache, act_cache = cache

    if act == "relu":
        dZ = relu_backward(dA, act_cache)
        dA, dW, db = linear_backward(dZ, lin_cache)
    elif act == "sigmoid":
        dZ = sigmoid_backward(dA, act_cache)
        dA, dW, db = linear_backward(dZ, lin_cache)

    return dA, dW, db


def linear_act_backward(dA, cache, act):
    """
    Implements the linear and activation function derivatives of single node
    """
    lin_cache, act_cache = cache

    if act == "relu":
        dZ = relu_backward(dA, act_cache)
        dA, dW, db = linear_backward(dZ, lin_cache)
    elif act == "sigmoid":
        dZ = sigmoid_backward(dA, act_cache)
        dA, dW, db = linear_backward(dZ, lin_cache)

    return dA, dW, db


def forward_prop(X: np.ndarray, parameters: dict) -> tuple:
    """
    Implements the forward propagation function for whole neural network
    """
    # initialize parameters
    caches = []
    A = X
    L = len(parameters) // 2

    # propagate over hidden layers
    for l in range(1, L):
        # run the function on given node
        A, cache = linear_act_forward(A,
                                      parameters['W' + str(l)],
                                      parameters['b' + str(l)],
                                      "relu")

        # store its cache
        caches.append(cache)

    # propagate over last layer
    AL, cache = linear_act_forward(A,
                                   parameters['W' + str(L)],
                                   parameters['b' + str(L)],
                                   "sigmoid")
    caches.append(cache)

    # return final values
    return AL, caches


def backward_prop(AL: np.array, Y: np.array, caches: list) -> dict:
    """
    Implements the backward propagation function for whole neural network
    """
    # initialize variables
    gradients = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # backprop over the sigmoid function
    dAL = - ((Y / AL) - ((1 - Y)/ (1 - AL)))

    # backprop over linear function
    curr_cache = caches[-1]
    dA, dW, db = linear_act_backward(dAL, curr_cache, "sigmoid")
    gradients["dA" + str(L-1)] = dA
    gradients["dW" + str(L)] = dW
    gradients["db" + str(L)] = db

    # backprop over all other layers
    for l in reversed(range(L-1)):
        curr_cache = caches[l]
        dA, dW, db = linear_act_backward(gradients["dA" + str(l+1)],
                                         curr_cache,
                                         "relu")
        gradients["dA" + str(l)] = dA
        gradients["dW" + str(l+1)] = dW
        gradients["db" + str(l+1)] = db

    # return the computed gradients
    return gradients


def backward_prop_with_L2(AL: np.array, Y: np.array, caches: list, lamda: float) -> dict:
    """
    Implements the backward propagation function with L2 regularization
    """
    # initialize variables
    gradients = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # backprop over the sigmoid function
    dAL = - ((Y / AL) - ((1 - Y)/ (1 - AL)))

    # backprop over linear function
    curr_cache = caches[-1]
    dA, dW, db = linear_act_backward(dAL, curr_cache, "sigmoid")
    gradients["dA" + str(L-1)] = dA
    gradients["dW" + str(L)] = dW + (lamda * parameters['W' + str(L)]) / m
    gradients["db" + str(L)] = db

    # backprop over all other layers
    for l in reversed(range(L-1)):
        curr_cache = caches[l]
        dA, dW, db = linear_act_backward(gradients["dA" + str(l+1)],
                                         curr_cache,
                                         "relu")
        gradients["dA" + str(l)] = dA
        gradients["dW" + str(l+1)] = dW + (lamda * parameters['W' + str(l+1)]) / m
        gradients["db" + str(l+1)] = db

    # return the computed gradients
    return gradients


def forward_prop_with_dropout(X: np.ndarray, parameters: dict, kappa: float) -> tuple:
    """
    Implements the forward propagation function for dropout regularization
    """
    # initialize parameters
    caches = []
    dropouts = {}
    A = X
    L = len(parameters) // 2

    # propagate over hidden layers
    for l in range(1, L):
        # run the function on given node
        A, cache, D = linear_act_forward_with_dropout(A,
                                                   parameters['W' + str(l)],
                                                   parameters['b' + str(l)],
                                                   "relu",
                                                   kappa)
        # store its cache
        caches.append(cache)
        dropouts["D" + str(l)] = D

    # propagate over last layer
    AL, cache, D = linear_act_forward_with_dropout(A,
                                                parameters['W' + str(L)],
                                                parameters['b' + str(L)],
                                                "sigmoid",
                                                kappa)
    caches.append(cache)
    dropouts["D" + str(L)] = D

    # return final values
    return AL, caches, dropouts


def backward_prop_with_dropouts(AL: np.array, Y: np.array, caches: list, kappa: float, dropouts: dict) -> dict:
    """
    Implements the backward propagation function for whole neural network
    """
    # initialize variables
    gradients = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # backprop over the sigmoid function
    dAL = - ((Y / AL) - ((1 - Y)/ (1 - AL)))

    # backprop over linear function
    curr_cache = caches[-1]
    dA, dW, db = linear_act_backward(dAL, curr_cache, "sigmoid")
    gradients["dA" + str(L-1)] = (dA * dropouts['D' + str(L-1)]) / kappa
    gradients["dW" + str(L)] = dW
    gradients["db" + str(L)] = db

    # backprop over all other layers
    for l in reversed(range(L-1)):
        curr_cache = caches[l]
        dA, dW, db = linear_act_backward(gradients["dA" + str(l+1)],
                                         curr_cache,
                                         "relu")
        gradients["dA" + str(l)] = (dA * dropouts['D' + str(l)]) / kappa
        gradients["dW" + str(l+1)] = dW
        gradients["db" + str(l+1)] = db

    # return the computed gradients
    return gradients
