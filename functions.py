"""
This file contains all the subsidiary functions required for forward
prop and backprop algorithms in the other files
"""

import numpy as np
from activations import sigmoid, relu, sigmoid_backward, relu_backward


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
    elif act == "softmax":
        Z, lin_cache = linear(A, W, b)
        A, act_cache = softmax(Z)

    cache = (lin_cache, act_cache)

    return A, cache


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


def conv_zero_pad(x, pad):
    """
    Applies zero-padding to given matrix for convolutional network
    """
    return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=(0, 0))


def conv_mask(x):
    """
    Applies mask to given matrix for convolutional network
    """
    return x == np.max(x)


def conv_dist_val(dz, shape):
    """
    Applies the backward counterpart of mask for
    convolutional network backprop
    """
    nh, nw = shape
    return np.ones((nh, nw)) * dz / (nh * nw)


def conv_step(prev_slice, W, b):
    """
    Applies a single step of CNN foward prop
    """
    return np.sum(prev_slice * W) + float(b)


def recc_cell_forward(xt, a_prev, parameters):
    """
    Implements forward propagation in a single RNN unit
    """
    W_ax, W_aa, W_ya, b_a, b_y = parameters.values()

    a_next = np.tanh(np.dot(W_aa, a_prev) + np.dot(W_ax, xt) + b_a)
    yt_hat = softmax(np.dot(W_ya, a_next) + b_y)

    return a_next, yt_hat, (a_next, yt_hat, xt, parameters)


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implements forward propagation in a single LSTM unit
    """
    W_f, b_f, W_u, b_u, W_c, b_c, W_o, b_o, W_y, b_y = parameters.values()

    conc = np.concatenate((a_prev, xt), axis=0)

    gamma_f = sigmoid(np.dot(W_f, conc) + b_f)
    gamma_u = sigmoid(np.dot(W_u, conc) + b_u)
    gamma_ct = np.tanh(np.dot(W_c, conc) + b_c)
    c_next = gamma_f * c_prev + gamma_u * gamma_ct
    gamma_o = sigmoid(np.dot(W_o, conc) + b_o)
    a_next = gamma_o * np.tanh(c_next)
    yt_hat = softmax(np.dot(W_y, a_next) + b_y)

    return a_next, c_next, yt_hat, \
    (a_next, c_next, a_prev, c_prev, gamma_f, gamma_u, gamma_ct, gamma_o, xt, parameters)


def recc_cell_backward(da_next, cache):
    """
    Implements backward propagation in a single RNN unit
    """
    _, a_prev, xt, parameters = cache
    W_ax, W_aa, _, b_a, _ = parameters.values()

    dz = 1 - np.sqaure(np.tanh(np.dot(W_ax, xt) + np.dot(W_aa, a_prev) + b_a))
    dxt = np.dot(W_ax.T, (da_next * dz))
    dW_ax = np.dot((da_next * dz), xt.T)
    da_prev = np.dot(W_aa.T, (da_next * dz))
    dW_aa = np.dot((da_next * dz), a_prev.T)
    db_a = np.sum((da_next * dz), axis=1, keepdims=True)

    return {
        "dxt": dxt,
        "da_prev": da_prev,
        "dW_ax": dW_ax,
        "dW_aa": dW_aa,
        "db_a": db_a
    }


def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implements backward propagation in a single LSTM unit
    """
    a_next, c_next, a_prev, c_prev, gamma_f, gamma_u, gamma_ct, gamma_o, xt, parameters = cache
    W_f, _, W_u, _, W_o, _, W_c, _, W_u, _ = parameters.values()
    n_a, _ = a_next.shape
    
    dtanh = 1 - np.square(np.tanh(c_next))
    dgamma_u = da_next * np.tanh(c_next) * gamma_o * (1 - gamma_o)
    dgamma_ct = (dc_next * gamma_u + gamma_o * dtanh * gamma_u * da_next) * (1 - gamma_ct**2)
    dgamma_o = (dc_next * gamma_ct + gamma_o * dtanh * gamma_ct * da_next) * gamma_u * (1 - gamma_u)
    dgamma_f = (dc_next * c_prev + gamma_o * dtanh * c_prev * da_next) * gamma_f * (1 - gamma_f)
    
    conc = np.concatenate((a_prev, xt), axis=0)
    dW_f = np.dot(dgamma_f, conc.T)
    dW_u = np.dot(dgamma_u, conc.T)
    dW_c = np.dot(dgamma_ct, conc.T)
    dW_o = np.dot(dgamma_o, conc.T)
    db_f = np.sum(dgamma_f, axis=1, keepdims=True)
    db_u = np.sum(dgamma_u, axis=1, keepdims=True)
    db_c = np.sum(dgamma_ct, axis=1, keepdims=True)
    db_o = np.sum(dgamma_o, axis=1, keepdims=True)

    da_prev = np.dot(W_f[:, :n_a].T, dgamma_f) + np.dot(W_u[:, :n_a].T, dgamma_u) + \
              np.dot(W_c[:, :n_a].T, dgamma_ct) + np.dot(W_o[:, :n_a].T, dgamma_o)
    dc_prev = dc_next * gamma_f + gamma_o * dtanh * gamma_f * da_next
    dxt = np.dot(W_f[:, n_a:].T, dgamma_f) + np.dot(W_u[:, n_a:].T, dgamma_u) + \
          np.dot(W_c[:, n_a:].T, dgamma_ct) + np.dot(W_o[:, n_a:].T, dgamma_o)
    
    return {
        "dxt": dxt,
        "da_prev": da_prev,
        "dc_prev": dc_prev,
        "dW_f": dW_f,
        "db_f": db_f,
        "dW_u": dW_u,
        "db_u": db_u,
        "dW_c": dW_c,
        "db_c": db_c,
        "dW_o": dW_o,
        "db_o": db_o
    }
