import numpy as np


def zero_pad(x, pad):
    return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=(0, 0))


def mask(x):
    return x == np.max(x)


def dist_val(dz, shape):
    nh, nw = shape
    return np.ones((nh, nw)) * dz / (nh * nw)


def conv_step(prev_slice, W, b):
    return np.sum(prev_slice * W) + float(b)


def conv_forward_prop(A_prev, W, b, hpar):
    m, nh_prev, nw_prev, _ = A_prev.shape
    f, _, _, nc = W.shape
    stride = hpar["stride"]
    pad = hpar["pad"]

    nh = int((nh_prev - f + 2*pad) / stride) + 1
    nw = int((nw_prev - f + 2*pad) / stride) + 1

    Z = np.zeros((m, nh, nw, nc))

    A_prev_pad = zero_pad(A_prev, pad)

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


def pool_forward_prop(A_prev, hpar, mode="max"):
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


def conv_backward_prop(dZ, cache):
    A_prev, W, _, hpar = cache
    m, nh_prev, nw_prev, nc_prev = A_prev.shape
    f, _, _, nc = W.shape
    stride = hpar["stride"]
    pad = hpar["pad"]
    m, nh, nw, nc = dZ.shape

    dA_prev = np.zeros((m, nh_prev, nw_prev, nc_prev))
    dW = np.zeros((f, f, nc_prev, nc))
    db = np.zeros((1, 1, 1, nc))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

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


def pool_backward_prop(dA, cache, mode="max"):
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
                        mask = mask(a_slice)
                        dA_prev[i, v_st:v_nd, h_st:h_nd, c] += dA[i, h, w, c] * mask
                    elif mode == "avg":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, v_st:v_nd, h_st:h_nd, c] += dist_val(da, shape)
    
    return dA_prev
