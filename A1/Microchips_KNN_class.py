import numpy as np
import statistics

# EUCLIDIAN DISTANCE: for each vector component -> squared difference sqrt((x2 - x1)ˆ2 + (y2 - y1)ˆ2)
# x1 and x2 are arrays of 2 values respectively, the subtraction subracts the 0 position of both arrays and the 1 position of both arrays
# the 2 values are then squared, summed and the square root is taken -> the result is the units of distance!
def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KNN:
    k = None
    def __init__(self, k):
        self.k = k;

    # assign training data X (coordinates) and training labels y (1's and 0's)
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # capital X is a list containing all the attributes of a point (x1 value and x2 value of the microchips in this case)
    def predict(self, X):

        # every point in capital X is predicted against the training data
        predicted_lables = [self._predict(x) for x in X]

        return np.array(predicted_lables, dtype='f')



    # _predict takes a point and computes the distances between itself and the training data.
    # When all the distances from that point are computed, the indexes of the k-nearest points
    # are found with np.argsort. k_nearest_labels is a list containing labels (1 or 0) from y-train
    # that correspond to the shortest distances to the considered point.
    # Statistics.mode is the used to find the label that occurs more often.
    def _predict(self, x):
        
        # compute distances
        # find the distance between x and each value present in x_train
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        
        # get k nearest samples, labels
        # np.argsort() sorts an original list and returns the indexes of the values in the sorted list based on the values' indeces in the original list.
        k_indeces = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indeces]

        temp = []
        for label in k_nearest_labels:
            temp.append(label)
        
        # majority vote
        most_common = statistics.mode(temp)
        most_common = np.array([most_common])
        
        return most_common
    
    