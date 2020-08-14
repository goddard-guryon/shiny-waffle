from nn_imports import *


def quickWaffle(layers, X_train, y_train, num_epochs=10000, batch_size=64,
          print_cost=False, initialization="he", optimization="adam",
          regularization="none", alpha=0.5, beta=0.9, gamma=0.999, delta=0.0,
          epsilon=1e-8, lamda=0.5, kappa=0.5, softmax=False):
    """
    Before you start working on the network archtecture, you might
    want to just make a quick model and see how well it works.
    In that case, use this function to make a quick model\n
    :param layers: dimensions of neural network eg- [3, 5, 2] (list)
    ONLY ADD THE SIZES OF HIDDEN LAYERS, the first and last layer is added automatically\n
    :param `X_train`: training feature set (numpy ndarray)\n
    :param `y_train`: training labels (numpy array)\n
    :param `num_epochs`: iterations of model training (int)\n
    :param `batch_size`: size of mini-batch (int)\n
    :param `print_cost`: set to True to see training progress (bool)\n
    :param `initialization`: zero/random/he (str)\n
    :param `optimization`: gd/momentum/rmsprop/adam (str)\n
    :param `regularization`: none/L2 (str)\n
    :param `alpha`: gradient descent learning rate (float)\n
    :param `beta`: momentum parameter (float)\n
    :param `gamma`: RMSProp parameter (float)\n
    :param `delta`: learning rate decay parameter (float)\n
    :param `epsilon`: Adam parameter (int)\n
    :param `lamda`: L2 regularization parameter (float)\n
    :param `kappa`: dropout probability parameter (float)\n
    :param `softmax`: whether the model requires multi-class regression (bool)\n\n
    :returns: model parameters (dict) and costs (dict)
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
            AL, cache = linear_forward(X, parameters, softmax)

            # compute cost value
            if regularization == "L2":
                total_cost += cost_with_L2(AL, parameters, y, lamda)
            else:
                total_cost += cross_entropy_cost_mini(AL, y_train)

            # backward propagation
            if regularization == "L2":
                gradients = linear_backward_with_L2(AL, y, cache, lamda, parameters, softmax)
            else:
                gradients = linear_backward(AL, y, cache, softmax)

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

    # let the user know we're done with the training
    print(f"Finished cooking the waffle for {num_epochs} minutes\n")
    
    # let's show how the model looks
    if print_cost:
        network = DrawNN([20] + layers[1:])
        network.draw()

    # return the important stuff
    return parameters, costs
