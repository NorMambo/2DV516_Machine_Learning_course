import numpy as np
from sklearn.datasets import load_iris
import Functions as fs
import matplotlib.pyplot as plt

# NOTE: The program is very slow!!

iris = load_iris()

X = iris.data[:, :4]
y = iris.target

from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# scaler.fit(X)
# X = scaler.transform(X)

def sammon(X, iter, error_threshold, learing_rate):

    # This is how we can initialize y for the exercise. Gradient descent will improve these values.
    # Could be seen as betas (in previous exercises). These values should be improved with every iteration.
    np.random.seed(1)
    y = np.random.rand(X.shape[0], 2)
    y_next = np.zeros((X.shape[0], 2))

    # Will use the combination formula nCk and return an array of distances between all the points in y (low dimension).
    # Ex: if we have 4 points, it will return 6 distances (4 choose 2)
    dist_high_dim = fs.find_all_dist_combinations(X)

    # To avoid division by zero
    dist_high_dim += 1e-8

    # sum of all the DELTA_ij or distances in the high dimension (a constant value)
    sum_DELTA_ij = fs.sum_all_dist_combinations(X)

    # C is constant and equals sum_DELTA_ij
    c = sum_DELTA_ij

    z = 0
    while z < iter:

        # 1st iteration will take the random configuration of y. Then, the updated y is passed to the function iteratively.
        # Will use the combination formula nCk and return an array of distances between all the points in y (low dimension).
        # Ex: if we have 4 points, it will return 6 distances (4 choose 2)
        dist_low_dimension = fs.find_all_dist_combinations(y)

        # Sammon's stess can be found in Functions.py
        stress = fs.sammon_stress(sum_DELTA_ij, dist_high_dim, dist_low_dimension)
        print(stress)

        # if threshold is reached, stop and return y
        if stress <= error_threshold:
            return y

        else:
            
            par_der_1 = np.zeros(2)
            par_der_2 = np.zeros(2)

            # For every point in the dataset
            for i in range(X.shape[0]):

                # For every point in the dataset
                for j in range(X.shape[0]):

                    # Compute the following only i and j are not the same (we don't want divisions by zero)
                    if i != j:
                        
                        # DELTA_ij - distance between 2 single data points in high dim
                        dist_X_pts = fs.euclidean_dist(X[i], X[j])

                        # d_ij - distance between 2 single data points in low dim
                        dist_y_pts = fs.euclidean_dist(y[i], y[j])

                        # Product of d_ij and DELTA_ij
                        product = dist_X_pts * dist_y_pts
                        if (product == 0.0):
                            product = 0.1
                        
                        # Partial derivatives (According to Sammon's formula)
                        par_der_1 += ((dist_X_pts - dist_y_pts)/product) * (y[i] - y[j])
                        par_der_2 += (1/product) * ((dist_X_pts - dist_y_pts) - np.square(y[i] - y[j])/dist_y_pts * (1 + (dist_X_pts - dist_y_pts)/dist_y_pts))

                # Will dictate how much a point in the low dimension y[i] will move
                gradient = ((-2/c) * par_der_1) / (np.absolute((-2/c) * par_der_2))

                # This fills every position in the y_next array with modified y values (according to the gradient descent)
                y_next[i] = y[i] - learing_rate * gradient
            
            # The whole y array gets replaced with the new y array found with gradient descent
            y = np.copy(y_next)

        # Increase iteration number
        z+=1
        
    return y

X_pricted = sammon(X, 30, 0.2, 0.01)

plt.scatter(X_pricted[:, 0], X_pricted[:, 1], c=y, cmap='winter')
plt.show()

