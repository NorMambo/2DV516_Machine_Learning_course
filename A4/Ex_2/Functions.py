import numpy as np
from scipy.spatial.distance import pdist

# Calculate cost
def sammon_stress(sum_DELTA_ij, dist_high_dimension, dist_in_low_dim):
    stress = (1/sum_DELTA_ij) * (np.sum(np.square(dist_high_dimension - dist_in_low_dim) / dist_high_dimension))
    return stress

# Returns a single number
def sum_all_dist_combinations(any_dataset):
    sum = 0
    for i in range(any_dataset.shape[0]):

        for j in range(i + 1, any_dataset.shape[0]):

            # Eucliedean distance:
            dist = euclidean_dist(any_dataset[i], any_dataset[j])
            sum += dist
    return sum

# returns an array
def find_all_dist_combinations(any_dataset):
    empty = np.empty(shape=(0, 1))
    for i in range(any_dataset.shape[0]):

        for j in range(i + 1, any_dataset.shape[0]):
            
            # Euclidean distance
            dist = euclidean_dist(any_dataset[i], any_dataset[j])
            empty = np.vstack((empty, dist))

    # Could be flattened
    empty = empty.flatten()
    full = empty
    return full

# Find euclidean distance between 2 points
def euclidean_dist(pt1, pt2):
    dist = np.sqrt(np.sum(np.square(pt1 - pt2)))
    return dist