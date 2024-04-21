import numpy as np
from sklearn.cluster import KMeans

def bisecting_kmeans(X, k, iter):

    # where to store the result
    result_cluster = np.zeros(len(X), dtype=int)

    # initialize min_SSE
    min_SSE = 0

    # assign X to the variable that will be updated in the loops
    cluster_to_fit = X

    for i in range(1, k):
        for j in range(iter):

            # fit and find the SSE for the bisection
            k_means = KMeans(n_clusters=2, n_init=10)
            k_means.fit(cluster_to_fit)
            SSE = k_means.inertia_

            # find cluster indexes for the round
            if j == 0:
                min_SSE = SSE
                cluster_indexes = k_means.labels_
            else:
                if SSE < min_SSE:
                    min_SSE = SSE
                    cluster_indexes = k_means.labels_

        # create a copy of the result cluster where the result cluster has label zero
        # remaining_free_spots will always just be an array of zeros, but it will decrease in size
        remaining_free_spots = np.copy(result_cluster[result_cluster == 0])

        # create the subclusters
        subcluster1 = cluster_to_fit[cluster_indexes == 1]
        subcluster2 = cluster_to_fit[cluster_indexes == 0]

        # find SSE's
        sse1 = np.sum(np.square(subcluster1 - np.mean(subcluster1)))
        sse2 = np.sum(np.square(subcluster2 - np.mean(subcluster2)))

        # if SSE1 (of subcluster 1) is greater, fit Kmeans with subcluster1 in the next round
        if (sse1 > sse2):
            cluster_to_fit = subcluster1

            # let the label assigned to subcluster2 (lower SSE) take the spots in remaining_free_spots
            # note that the zeros will not be affected
            remaining_free_spots[cluster_indexes == 0] = i

        # if SSE2 (of subcluster 2) is greater, fit Kmeans with subcluster2 in the next round
        else:
            cluster_to_fit = subcluster2

            # let the label assigned to subcluster 1 (lower SSE) take the spots in remaining_free_spots
            remaining_free_spots[cluster_indexes == 1] = i

        # where result_cluster == 0, replace with the full layout of remaining_free_spots
        # remaining_free_spots and result_cluster[result_cluse == 0] have the same shape!
        result_cluster[result_cluster == 0] = remaining_free_spots


    return result_cluster

def sammon(X, iter, error_threshold, learing_rate):

    # This is how we can initialize y for the exercise. Gradient descent will improve these values.
    # Could be seen as betas (in previous exercises). These values should be improved with every iteration.
    np.random.seed(1)
    y = np.random.rand(X.shape[0], 2)
    y_next = np.zeros((X.shape[0], 2))

    # Will use the combination formula nCk and return an array of distances between all the points in y (low dimension).
    # Ex: if we have 4 points, it will return 6 distances (4 choose 2)
    dist_high_dim = find_all_dist_combinations(X)

    # To avoid division by 0
    dist_high_dim += 1e-8

    # sum of all the DELTA_ij or distances in the high dimension (a constant value)
    sum_DELTA_ij = sum_all_dist_combinations(X)

    # C is constant and equals sum_DELTA_ij
    c = sum_DELTA_ij

    z = 0
    while z < iter:

        # 1st iteration will take the random configuration of y. Then, the updated y is passed to the function iteratively.
        # Will use the combination formula nCk and return an array of distances between all the points in y (low dimension).
        # Ex: if we have 4 points, it will return 6 distances (4 choose 2)
        dist_low_dimension = find_all_dist_combinations(y)

        # Sammon's stess can be found in Functions.py
        stress = sammon_stress(sum_DELTA_ij, dist_high_dim, dist_low_dimension)
        # print(stress)

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
                        dist_X_pts = euclidean_dist(X[i], X[j])

                        # d_ij - distance between 2 single data points in low dim
                        dist_y_pts = euclidean_dist(y[i], y[j])

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

# Calculate cost
def sammon_stress(DELTA_ij, dist_high_dimension, dist_in_low_dim):
    stress = (1/DELTA_ij) * (np.sum(np.square(dist_high_dimension - dist_in_low_dim) / dist_high_dimension))
    return stress

# Returns a single number
def sum_all_dist_combinations(any_dataset):
    sum = 0
    for i in range(any_dataset.shape[0]):

        for j in range(i + 1, any_dataset.shape[0]):

            # Eucliedean distance:
            dist = euclidean_dist(any_dataset[i], any_dataset[j])
            # dist = np.sqrt(np.sum(np.square(any_dataset[i] - any_dataset[j])))
            sum += dist
    return sum

# returns an array
def find_all_dist_combinations(any_dataset):
    empty = np.empty(shape=(0, 1))
    for i in range(any_dataset.shape[0]):

        for j in range(i + 1, any_dataset.shape[0]):
            
            # Euclidean distance
            dist = euclidean_dist(any_dataset[i], any_dataset[j])
            # dist = np.sqrt(np.sum(np.square(any_dataset[i] - any_dataset[j])))
            empty = np.vstack((empty, dist))

    # Could be flattened
    empty = empty.flatten()
    full = empty
    return full

# Find euclidean distance between 2 points
def euclidean_dist(pt1, pt2):
    dist = np.sqrt(np.sum(np.square(pt1 - pt2)))
    return dist