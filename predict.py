import numpy as np
from linear import lin_forward_prop
from metrics import accuracy_score


def taste_waffle(X, y, parameters, softmax=False):
    """
    Use your model to make predictions on given data
    """
    preds = np.zeros((1, X.shape[1]), dtype=np.int)
    AL, _ = lin_forward_prop(X, parameters, softmax)
    for i in range(AL.shape[1]):
        if AL[0, i] > 0.5:
            preds[0, i] = 1
        else:
            preds[0, i] = 0
    
    print(f"Your waffle tastes {accuracy_score(preds, y):.3f}% yummy!")
    return preds
