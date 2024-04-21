import numpy as np

def normal_equation(X, y):

    W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return W

def extend(X, degree):
    Xe = np.ones((X.shape[0], 1))

    for i in range(1, degree + 1):
        new_column = X**i
        Xe = np.c_[Xe, new_column]

    return Xe