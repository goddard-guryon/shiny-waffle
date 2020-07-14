import numpy as np


def accuracy_score(y_hat: np.array, y: np.array) -> float:
    return 100 - np.mean(np.abs(y_hat - y)) * 100
