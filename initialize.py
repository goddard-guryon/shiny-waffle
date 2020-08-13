import numpy as np
import math


def initialize_zeros(layers: np.array) -> dict:
    """
    Initializes parameters as arrays of zeros
    """
    # initialize variables
    parameters = {}

    # create parameters
    for l in range(1, len(layers)):
        parameters['W'] = np.zeros((layers[l], layers[l-1]))
        parameters['b'] = np.zeros((layers[l], 1))

    return parameters


def initialize_random(layers: np.array) -> dict:
    """
    Initializes parameters as arrays of small random values and
    zeros respectively
    """
    # initialize variables
    parameters = {}

    # start creating parameters
    for l in range(1, len(layers)):
        parameters['W'] = np.random.randn(layers[l], layers[l-1]) * .01
        parameters['b'] = np.zeros((layers[l], 1))

    return parameters


def initialize_he(layers: np.array) -> dict:
    """
    Initializes parameters as arrays containing values according to
    He Initialization method, specifically, multiplies the initial random
    values by the sqaure root of 2 / previous layer
    """
    # initialize variables
    parameters = {}

    # start creating parameters
    for l in range(1, len(layers)):
        x = np.sqrt(2. / layers[l-1])
        parameters['W'] = np.random.randn(layers[l], layers[l-1]) * x
        parameters['b'] = np.zeros((layers[l], 1))

    return parameters


def mini_batches(X, Y, batch_size=64) -> list:
    """
    Create mini-batches for mini-batch gradient descent
    """
    m = X.shape[1]
    batches = []

    shuffle = list(np.random.permutation(m))
    shuffled_X = X[:, shuffle]
    shuffled_Y = Y[:, shuffle].reshape((1, m))

    complete_batches = math.floor(m / batch_size)
    for b in range(complete_batches):
        mini_x = shuffled_X[:, batch_size*b:batch_size*(b+1)]
        mini_y = shuffled_Y[:, batch_size*b:batch_size*(b+1)]
        batches.append((mini_x, mini_y))

    if m % batch_size != 0:
        mini_x = shuffled_X[:, -(m-batch_size*complete_batches):]
        mini_y = shuffled_Y[:, -(m-batch_size*complete_batches):]
        batches.append((mini_x, mini_y))

    return batches


def initialize_velocity(parameters: dict) -> dict:
    """
    Initialize velocity parameter for gradient descent with momentum
    """
    velocity = {}
    L = len(parameters) // 2

    for l in range(L):
        velocity["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        velocity["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

    return velocity


def initialize_rms(parameters: dict) -> dict:
    """
    Initialize RMS parameter for RMSProp optimization
    """
    L = len(parameters) // 2
    rms = {}

    for l in range(L):
        rms["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        rms["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

    return rms

def initialize_adam(parameters: dict) -> tuple:
    """
    Initialize parameters for Adam optimization algorithm
    """
    L = len(parameters) // 2
    velocity = {}
    rms = {}

    for l in range(L):
        velocity["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        velocity["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
        rms["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        rms["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

    return velocity, rms
