"""
This file contains forward prop functions for dense, convolutional,
pooling, reccurent and LSTM layers
"""

import numpy as np
from functions import linear_act_forward, linear_act_backward, conv_zero_pad, conv_step,\
    recc_cell_forward, recc_cell_backward, lstm_cell_forward, lstm_cell_backward


def linear_forward(X: np.ndarray, parameters: dict, softmax: bool = False) -> tuple:
    """
    Implements the forward propagation function for a dense layer
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
    if softmax:
        AL, cache = linear_act_forward(A,
                                   parameters['W' + str(L)],
                                   parameters['b' + str(L)],
                                   "softmax")
    else:
        AL, cache = linear_act_forward(A,
                                   parameters['W' + str(L)],
                                   parameters['b' + str(L)],
                                   "sigmoid")
    caches.append(cache)

    # return final values
    return AL, caches


def conv_forward(A_prev, W, b, hpar):
    m, nh_prev, nw_prev, _ = A_prev.shape
    f, _, _, nc = W.shape
    stride = hpar["stride"]
    pad = hpar["pad"]

    nh = int((nh_prev - f + 2*pad) / stride) + 1
    nw = int((nw_prev - f + 2*pad) / stride) + 1

    Z = np.zeros((m, nh, nw, nc))

    A_prev_pad = conv_zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(nh):
            for w in range(nw):
                for c in range(nc):
                    v_st = h * stride
                    v_nd = v_st + f
                    h_st = w * stride
                    h_nd = h_st + f
                    a_slice = a_prev_pad[v_st:v_nd, h_st:h_nd, :]
                    weights = W[:, :, :, c]
                    bias = b[:, :, :, c]
                    Z[i, h, w, c] = conv_step(a_slice, weights, bias)
    
    return Z, (A_prev, W, b, hpar)


def pool_forward(A_prev, hpar, mode="max"):
    m, nh_prev, nw_prev, nc_prev = A_prev.shape
    f = hpar['f']
    stride = hpar["stride"]

    nh = int((nh_prev - f) / stride) + 1
    nw = int((nw_prev - f) / stride) + 1
    nc = nc_prev

    A = np.zeros((m, nh, nw, nc))

    for i in range(m):
        for h in range(nh):
            for w in range(nw):
                for c in range(nc):
                    v_st = h * stride
                    v_nd = v_st + f
                    h_st = w * stride
                    h_nd = h_st + f
                    a_slice = A_prev[i, v_st:v_nd, h_st:h_nd, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice)
                    elif mode == "avg":
                        A[i, h, w, c] = np.mean(a_slice)
    
    return A, (A_prev, hpar)


def recc_forward(x, a0, parameters):
    """
    Implements forward propagation for entire RNN
    """
    caches = []
    _, m, T_x = x.shape
    n_y, n_a = parameters["W_ya"].shape

    a = np.zeros((n_a, m, T_x))
    y_hat = np.zeros((n_y, m, T_x))
    a_next = a0

    for t in range(T_x):
        xt = x[:, :, t]
        a_next, yt_hat, cache = recc_cell_forward(xt, a_next, parameters)
        a[:, :, t] = a_next
        y_hat[:, :, t] = yt_hat
        caches.append(cache)
    
    return a, y_hat, (caches, x)


def lstm_forward(x, a0, parameters):
    """
    Implements forward propagation for an entire LSTM based RNN
    """
    caches = []
    _, m, T_x = x.shape
    n_y, n_a = parameters["W_y"].shape

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    a_next = a0
    c_next = np.zeros((n_a, m))

    for t in range(T_x):
        xt = x[:, :, t]
        a_next, c_next, y_hat, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = y_hat
        caches.append(cache)
    
    return a, y, c, (caches, x)
