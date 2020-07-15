import numpy as np


def update_parameters(parameters: dict, gradients: dict, alpha: float) -> dict:
    """
    Update network parameters to follow gradient descent
    """
    # initialize variables
    L = len(parameters) // 2

    # update
    for l in range(L):
        parameters['W' + str(l+1)] -= alpha * gradients["dW" + str(l+1)]
        parameters['b' + str(l+1)] -= alpha * gradients["db" + str(l+1)]

    return parameters


def update_parameters_with_momentum(parameters: dict, gradients: dict,
                                    velocity: dict, alpha: float, beta: float) -> dict:
    """
    Update network parameters to follow gradient descent with momentum
    """
    L = len(parameters) // 2

    for l in range(L):
        velocity["dW" + str(l+1)] = beta*velocity["dW" + str(l+1)] + (1-beta)*gradients["dW" + str(l+1)]
        velocity["db" + str(l+1)] = beta*velocity["db" + str(l+1)] + (1-beta)*gradients["db" + str(l+1)]
        parameters['W' + str(l+1)] -= alpha*velocity["dW" + str(l+1)]
        parameters['b' + str(l+1)] -= alpha*velocity["db" + str(l+1)]

    return parameters, velocity


def update_parameters_with_rms(parameters: dict, gradients: dict, rms: dict,
                               alpha: float, gamma: float) -> dict:
    """
    Update network parameters to follow gradient descent with RMSProp
    """
    L = len(parameters) // 2

    for l in range(L):
        rms["dW" + str(l+1)] = gamma*rms["dW" + str(l+1)] + (1-gamma)*gradients["dW" + str(l+1)]**2
        rms["db" + str(l+1)] = gamma*rms["db" + str(l+1)] + (1-gamma)*gradients["db" + str(l+1)]**2
        parameters['W' + str(l+1)] -= alpha*gradients["dW" + str(l+1)]/np.sqrt(rms["dW" + str(l+1)])
        parameters['b' + str(l+1)] -= alpha*gradients["db" + str(l+1)]/np.sqrt(rms["db" + str(l+1)])

    return parameters, rms


def update_parameters_with_adam(parameters: dict, gradients: dict, velocity: dict,
                                rms: dict, alpha: float, beta: float, gamma: float,
                                epsilon: float, t: int) -> dict:
    """
    Update network parameters to follow gradient descent with Adam optimization
    """
    L = len(parameters) // 2
    velocity_cor = {}
    rms_cor = {}

    for l in range(L):
        velocity["dW" + str(l+1)] = beta*velocity["dW" + str(l+1)] + (1-beta)*gradients["dW" + str(l+1)]
        velocity["db" + str(l+1)] = beta*velocity["db" + str(l+1)] + (1-beta)*gradients["db" + str(l+1)]

        velocity_cor["dW" + str(l+1)] = velocity["dW" + str(l+1)] / (1 - beta**t)
        velocity_cor["db" + str(l+1)] = velocity["db" + str(l+1)] / (1 - beta**t)

        rms["dW" + str(l+1)] = gamma*rms["dW" + str(l+1)] + (1-gamma)*gradients["dW" + str(l+1)]**2
        rms["db" + str(l+1)] = gamma*rms["db" + str(l+1)] + (1-gamma)*gradients["db" + str(l+1)]**2

        rms_cor["dW" + str(l+1)] = rms["dW" + str(l+1)] / (1 - gamma**t)
        rms_cor["db" + str(l+1)] = rms["db" + str(l+1)] / (1 - gamma**t)

        parameters['W' + str(l+1)] -= alpha*velocity_cor["dW" + str(l+1)]/(np.sqrt(rms_cor["dW" + str(l+1)] + epsilon))
        parameters['b' + str(l+1)] -= alpha*velocity_cor["db" + str(l+1)]/(np.sqrt(rms_cor["db" + str(l+1)] + epsilon))

    return parameters, velocity, rms
