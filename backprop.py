"""
This file contains all the backpropagation functions for dense,
convolutional, pooling, reccurent and LSTM layers
"""

import numpy as np
from functions import linear_act_backward, conv_zero_pad, conv_mask, conv_dist_val,\
    recc_cell_backward, lstm_cell_backward


def linear_backward(AL: np.array, Y: np.array, caches: list, softmax: bool) -> dict:
    """
    Implements the backward propagation function for whole neural network
    """
    # initialize variables
    gradients = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # backprop over the sigmoid function
    if softmax:
        dAL = AL - Y
    else:
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


def linear_backward_with_L2(AL: np.array, Y: np.array, caches: list, lamda: float, parameters, softmax: bool) -> dict:
    """
    Implements the backward propagation function with L2 regularization
    """
    # initialize variables
    gradients = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # backprop over the sigmoid function
    if softmax:
        dAL = AL - Y
    else:
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


def conv_backward(dZ, cache):
    A_prev, W, _, hpar = cache
    m, nh_prev, nw_prev, nc_prev = A_prev.shape
    f, _, _, nc = W.shape
    stride = hpar["stride"]
    pad = hpar["pad"]
    m, nh, nw, nc = dZ.shape

    dA_prev = np.zeros((m, nh_prev, nw_prev, nc_prev))
    dW = np.zeros((f, f, nc_prev, nc))
    db = np.zeros((1, 1, 1, nc))

    A_prev_pad = conv_zero_pad(A_prev, pad)
    dA_prev_pad = conv_zero_pad(dA_prev, pad)

    for i in range(m):
        a_pad = A_prev_pad[i]
        da_pad = dA_prev_pad[i]
        for h in range(nw):
            for w in range(nh):
                for c in range(nc):
                    v_st = h * stride
                    v_nd = v_st + f
                    h_st = w * stride
                    h_nd = h_st + f

                    a_slice = a_pad[v_st:v_nd, h_st:h_nd, :]

                    da_pad[v_st:v_nd, h_st:h_nd, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_pad[pad:-pad, pad:-pad, :]
    
    return dA_prev, dW, db


def pool_backward(dA, cache, mode="max"):
    A_prev, hpar = cache
    f = hpar['f']
    m, nh, nw, nc = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(nh):
            for w in range(nw):
                for c in range(nc):
                    v_st = h
                    v_nd = v_st + f
                    h_st = w
                    h_nd = h_st + f

                    if mode == "max":
                        a_slice = a_prev[v_st:v_nd, h_st:h_nd, c]
                        mask = conv_mask(a_slice)
                        dA_prev[i, v_st:v_nd, h_st:h_nd, c] += dA[i, h, w, c] * mask
                    elif mode == "avg":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, v_st:v_nd, h_st:h_nd, c] += conv_dist_val(da, shape)
    
    return dA_prev


def recc_backward(da, caches):
    """
    Implements backward propagation for entire RNN
    """
    caches, _ = caches
    x1 = caches[0][2]
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    dx = np.zeros((n_x, m, T_x))
    dW_ax = np.zeros((n_a, n_x))
    dW_aa = np.zeros((n_a, n_a))
    db_a = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prev = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        gradients = recc_cell_backward(da[:, :, t] + da_prev, caches[t])
        dx[:, :, t] = gradients["dxt"]
        dW_ax += gradients["dW_ax"]
        dW_aa += gradients["dW_aa"]
        db_a += gradients["db_a"]
        da_prev = gradients["da_prev"]
    da0 = da_prev

    return {
        "dx": dx,
        "da0": da0,
        "dW_ax": dW_ax,
        "dW_aa": dW_aa,
        "db_a": db_a
    }


def lstm_backward(da, caches):
    """
    Implements backward propagation for an entire LSTM based RNN
    """
    caches, _ = caches
    x1 = caches[0][9]
    n_a, m, T_x = da.shape
    n_x, _ = x1.shape

    dx = np.zeros((n_x, m, T_x))
    da_prev = np.zeros((n_a, m))
    dc_prev = np.zeros((n_a, m))
    dW_f = np.zeros((n_a, n_a + n_x))
    dW_u = np.zeros((n_a, n_a + n_x))
    dW_c = np.zeros((n_a, n_a + n_x))
    dW_o = np.zeros((n_a, n_a + n_x))
    db_f = np.zeros((n_a, 1))
    db_u = np.zeros((n_a, 1))
    db_c = np.zeros((n_a, 1))
    db_o = np.zeros((n_a, 1))

    for t in reversed(range(T_x)):
        gradients = lstm_cell_backward(da[:, :, t] + da_prev, dc_prev, caches[t])
        da_prev, dc_prev, dx[:, :, t] = gradients["da_prev"], gradients["dc_prev"], gradients["dxt"]
        dW_f += gradients["dW_f"]
        dW_u += gradients["dW_u"]
        dW_c += gradients["dW_c"]
        dW_o += gradients["dW_o"]
        db_f += gradients["db_f"]
        db_u += gradients["db_u"]
        db_c += gradients["db_c"]
        db_o += gradients["db_o"]

    return {
        "dx": dx,
        "da0": gradients["da_prev"],
        "dW_f": dW_f,
        "db_f": db_f,
        "dW_u": dW_u,
        "db_i": db_u,
        "dW_c": dW_c,
        "db_c": db_c,
        "dW_o": dW_o,
        "db_o": db_o
    }
