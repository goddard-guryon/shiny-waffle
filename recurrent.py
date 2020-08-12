from activations import softmax, sigmoid
import numpy as np


def rnn_cell_forward_prop(xt, a_prev, parameters):
    """
    Implements forward propagation in a single RNN unit
    """
    W_ax, W_aa, W_ya, b_a, b_y = parameters.values()

    a_next = np.tanh(np.dot(W_aa, a_prev) + np.dot(W_ax, xt) + b_a)
    yt_hat = softmax(np.dot(W_ya, a_next) + b_y)

    return a_next, yt_hat, (a_next, yt_hat, xt, parameters)


def lstm_cell_forward_prop(xt, a_prev, c_prev, parameters):
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


def rnn_cell_backward_prop(da_next, cache):
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


def lstm_cell_backward_prop(da_next, dc_next, cache):
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


def rnn_forward_prop(x, a0, parameters):
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
        a_next, yt_hat, cache = rnn_cell_forward_prop(xt, a_next, parameters)
        a[:, :, t] = a_next
        y_hat[:, :, t] = yt_hat
        caches.append(cache)
    
    return a, y_hat, (caches, x)


def lstm_forward_prop(x, a0, parameters):
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
        a_next, c_next, y_hat, cache = lstm_cell_forward_prop(xt, a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = y_hat
        caches.append(cache)
    
    return a, y, c, (caches, x)


def rnn_backward_prop(da, caches):
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
        gradients = rnn_cell_backward_prop(da[:, :, t] + da_prev, caches[t])
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


def lstm_backward_prop(da, caches):
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
        gradients = lstm_cell_backward_prop(da[:, :, t] + da_prev, dc_prev, caches[t])
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
