import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as col
from initialize import mini_batches, initialize_adam, initialize_he, initialize_random, initialize_rms, initialize_velocity, initialize_zeros
from propagate import forward_prop, backward_prop, backward_prop_with_L2
from cost import cross_entropy_cost_mini, cost_with_L2
from update import update_parameters, update_parameters_with_momentum, update_parameters_with_rms, update_parameters_with_adam
from draw import draw_neural_net


def make_waffle(layers: list, X_train: np.ndarray, y_train: np.ndarray,
          num_epochs: int = 10000, batch_size: int = 64, print_cost: bool = False,
          initialization: str = "he", optimization: str = "adam",
          regularization: str = "none", alpha: float = 0.5, beta: float = 0.9,
          gamma: float = 0.999, delta: float = 0.0, epsilon: int = 1e-8,
          lamda: float = 0.5, kappa: float = 0.5, softmax: bool = False):
    """
    Initialize neural network model and fit to given data
    :param layers: dimensions of neural network eg- [3, 5, 2] (list)
    ONLY ADD THE SIZES OF HIDDEN LAYERS, the first and last layer is added automatically
    :param X_train: training feature set (numpy ndarray)
    :param y_train: training labels (numpy array)
    :param num_epochs: iterations of model training (int)
    :param batch_size: size of mini-batch (int)
    :param print_cost: set to True to see training progress (bool)
    :param initialization: zero/random/he (str)
    :param optimization: gd/momentum/rmsprop/adam (str)
    :param regularization: none/L2 (str)
    :param alpha: gradient descent learning rate (float)
    :param beta: momentum parameter (float)
    :param gamma: RMSProp parameter (float)
    :param delta: learning rate decay parameter (float)
    :param epsilon: Adam parameter (int)
    :param lamda: L2 regularization parameter (float)
    :param kappa: dropout probability parameter (float)
    :param softmax: whether the model requires multi-class regression (bool)
    """
    # initialize parameters
    costs = []
    t = 0
    layers = [X_train.shape[0]] + layers + [1]
    m = X_train.shape[1]

    if initialization == "random":
        parameters = initialize_random(layers)
    elif initialization == "he":
        parameters = initialize_he(layers)
    else:
        parameters = initialize_zeros(layers)


    if optimization == "momentum":
        velocity = initialize_velocity(parameters)
    elif optimization == "rmsprop":
        rms = initialize_rms(parameters)
    elif optimization == "adam":
        velocity, rms = initialize_adam(parameters)
    else:
        pass

    print(f"Preparing to cook a waffle for {num_epochs} minutes...")


    # start iteration
    for epoch in range(num_epochs):

        # create the mini-batches
        batches = mini_batches(X_train, y_train, batch_size)
        total_cost = 0

        # implement learning rate decay if delta is specified by the user
        if delta > 0.0:
            alpha_0 = alpha * delta**epoch

        for batch in batches:
            X, y = batch

            # forward propagation
            AL, cache = forward_prop(X, parameters, softmax)

            # compute cost value
            if regularization == "L2":
                total_cost += cost_with_L2(AL, parameters, y, lamda)
            else:
                total_cost += cross_entropy_cost_mini(AL, y_train)

            # backward propagation
            if regularization == "L2":
                gradients = backward_prop_with_L2(AL, y, cache, lamda, parameters, softmax)
            else:
                gradients = backward_prop(AL, y, cache, softmax)

            # update parameter values
            if optimization == "momentum":
                parameters, velocity = update_parameters_with_momentum(parameters, gradients,
                                                             velocity, alpha_0, beta)
            elif optimization == "rmsprop":
                parameters, rms = update_parameters_with_rms(parameters, gradients,
                                                        rms, alpha_0, gamma)
            elif optimization == "adam":
                t += 1
                parameters, velocity, rms = update_parameters_with_adam(parameters, gradients,
                                                         velocity, rms, alpha_0,
                                                         beta, gamma, epsilon, t)
            else:
                parameters = update_parameters(parameters, gradients, alpha_0)

        # check average cost
        avg_cost = total_cost / m

        # print out some updates
        if epoch % 1000 == 0:
            costs.append(avg_cost)
            if print_cost:
                print(f"Waffle texture after {epoch} minutes: {avg_cost}")

    # let's show how the model looks
    if print_cost:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        draw_neural_net(ax, 0.05, 0.95, 1, 0, layers)
        plt.title(f"Your {len(layers)}-layer NN waffle (^ᴗ^)")
        plt.show()

    # return the important stuff
    return parameters, costs
