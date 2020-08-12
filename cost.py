import numpy as np

def cross_entropy_cost(AL: np.array, Y: np.array) -> float:
    """
    Apply cross-entropy cost function
    """
    m = Y.shape[1]
    cost = - np.sum(
                    np.dot(Y, np.log(AL).T) +
                    np.dot((1 - Y), np.log(1 - AL).T)) / m

    # make sure dimensions are correct
    return np.squeeze(cost)


def cross_entropy_cost_mini(AL: np.array, Y: np.array) -> float:
    """
    Apply cross-entropy cost for mini-batch gradient desccent
    """
    return np.sum(np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1-AL), (1-Y)))


def cost_with_L2(AL: np.array, parameters: dict, Y: np.array, lamda: float) -> float:
    """
    Apply cross-entropy cost with L2 regularization
    """
    m = Y.shape[1]
    L = len(parameters) // 2
    buffer = 0

    for l in range(L):
        buffer += np.sum(np.square(parameters['W' + str(l+1)]))

    cost = cross_entropy_cost_mini(AL, Y) + (lamda * buffer / (2 * m))

    return cost


def softmax_cost(AL: np.array, Y: np.array) -> np.array:
    """
    """
    return - np.sum(np.dot(Y, np.log(AL).T))
