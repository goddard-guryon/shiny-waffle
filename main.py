import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt
from initialize import *
from propagate import *
from cost import *
from update import *
from predict import *
from metrics import *


def my_nn(layers: list, X_train: np.ndarray, y_train: np.ndarray,
          num_epochs: int = 10000, batch_size: int = 64, print_cost: bool = False,
          initialization: str = "he", optimization: str = "adam",
          regularization: str = "none", alpha: float = 0.5, beta: float = 0.9,
          gamma: float = 0.999, epsilon: int = 1e-8, lamda: float = 0.0,
          kappa: float = 0.5):
    """
    Initialize neural network model and fit to given data
    :param layers: dimensions of neural network eg- [3, 5, 2] (list)
    :param X_train: training feature set (numpy ndarray)
    :param y_train: training labels (numpy array)
    :param num_epochs: iterations of model training (int)
    :param batch_size: size of mini-batch (int)
    :param print_cost: set to True to see training progress (bool)
    :param initialization: zero/random/he (str)
    :param optimization: gd/momentum/rmsprop/adam (str)
    :param regularization: none/L2/dropout (str)
    :param alpha: gradient descent learning rate (float)
    :param beta: momentum parameter (float)
    :param gamma: RMSProp parameter (float)
    :param epsilon: Adam parameter (int)
    :param lamda: L2 regulrization parameter (float)
    :param kappa: dropout probability parameter (float)
    """
    # initialize parameters
    costs = []
    A = X_train
    tau = 0
    layers = [X_train.shape[0]] + layers + [1]

    if initialization == "random":
        parameters = initialize_random(layers)
    elif initialization == "he":
        parameters = initialize_he(layers)
    else:
        parameters = initialize_zeros(layers)

    L = len(parameters) // 2

    if optimization == "momentum":
        velocity = initialize_velocity(parameters)
    elif optimization == "rmsprop":
        rms = initialize_rms(parameters)
    elif optimization == "adam":
        velocity, rms = initialize_adam(parameters)
    else:
        pass


    # start iteration
    for epoch in range(num_epochs):

        batches = mini_batches(X_train, y_train, batch_size)
        total_cost = 0

        for batch in batches:
            X, y = batch
            # forward propagation
            if regularization == "dropout":
                AL, cache, dropouts =forward_prop_with_dropout(X, parameters, kappa)
            else:
                AL, cache = forward_prop(X, parameters)

            # compute cost value
            if regularization == "L2":
                total_cost += cost_with_L2(AL, parameters, y, lamda)
            else:
                cost = cross_entropy_cost_mini(AL, y_train)

            # backward propagation
            if regularization == "L2":
                gradients = backward_prop_with_L2(AL, y, cache, lamda)
            elif regularization == "dropout":
                gradients = backward_prop_with_dropout(AL, y, cache, kappa, dropouts)
            else:
                gradients = backward_prop(AL, y, cache)

            # update parameter values
            if optimization == "momentum":
                parameters = update_parameters_with_momentum(parameters, gradients,
                                                             velocity, alpha, beta)
            elif optimization == "rmsprop":
                parameters = update_parameters_with_rms(parameters, gradients,
                                                        rms, alpha, gamma)
            elif optimization == "adam":
                parameters = update_parameters_with_adam(parameters, gradients,
                                                         velocity, rms, alpha,
                                                         beta, gamma, epsilon, tau)
            else:
                parameters = update_parameters(parameters, gradients, alpha)

            # check average cost
            avg_cost = total_cost / m

            # print out some updates
            if iter % 1000 == 0:
                costs.append(avg_cost)
                if print_cost:
                    print(f"Cost after {epoch}th epoch: {avg_cost}")

    # plot the cost function's values
    plt.plot(costs)
    plt.xlabel("Epoch")
    plt.ylabel("Cost Value")
    plt.title(f"Learning Rate = {alpha}")
    plt.show()

    # return the important stuff
    return parameters

import h5py


def load_dataset():
    loc = "/home/vihangbodh/musical-broccoli/course_1/Week 2/Logistic Regression as a Neural Network/"
    train_dataset = h5py.File(loc+'datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(loc+'datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T / 255
test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T / 255

model = my_nn([train_set_x.shape[0], 3, 4, 1], train_set_x, train_set_y, alpha=0.005, print_cost=True)
