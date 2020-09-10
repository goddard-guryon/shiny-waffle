"""
This file contains codes for data processing
"""


import numpy as np


def normalize(x):
    """
    Normalize the dataset by subtracting mean and dividing by
    standard deviation in the data
    """
    # compute mean and std
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # ensure we don't get NaN values
    std[std==0] = 1.

    # return normalized data
    return (x - mean) / std


def PCA(x, lim):
    """
    Perform Principal Component Anaylysis on the given dataset to
    reduce its dimensionality to {lim} dimensions
    """
    # normalize the data
    x = normalize(x)

    # make the covariance matrix of dataset
    cov = np.cov(x, rowvar=False, bias=True)

    # compute the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(cov)

    # we only need one eigenvector which will be our basis vector
    eig_vec = eig_vec[:, eig_val.argsort()[::-1]]

    # select only the first few of these basis vectors
    B = eig_vec[:, :lim]

    # compute the projection matrix
    P = B @ np.linalg.inv(B.T @ B) @ B.T

    # compute the dataset projection in reduced dimensions

    x = (P @ X.T).T
    
    # return the real part of it (numpy computation projects the values
    # onto the complex plane)
    return np.real(x)
