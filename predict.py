import numpy as np
from propagate import forward_prop
from metrics import accuracy_score


def predict(X, y, parameters):
    """
    """
    preds = np.zeros((1, X.shape[1]), dtype=np.int)
    AL, _ = forward_prop(X, parameters)
    for i in range(AL.shape[1]):
        if AL[0, i] > 0.5:
            preds[0, i] = 1
        else:
            preds[0, i] = 0
    
    print(f"Given waffle made prediction with {accuracy_score(preds, y):.3f}% accuracy")
    return preds
