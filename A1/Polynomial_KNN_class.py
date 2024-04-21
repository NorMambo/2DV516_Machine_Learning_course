import numpy as np

class KNN_regression:
    k = None
    def __init__(self, k):
        self.k = k

    # assign a training set of data
    def fit(self, X):
        self.X_train = X

    # pass a test set of data and process it
    def predict(self, X_test):
        list_of_points_with_respective_predicted_y_coordinate = []
        mean_error_sum = 0
        for point_x_y in X_test:
            
            # predict the nearest y-coordinate
            mean_y_value = self.distances_and_mean_distance(point_x_y, self.k)

            # point_x_y[0] is the x value of the point
            list_of_points_with_respective_predicted_y_coordinate.append((point_x_y[0], mean_y_value))
            mean_error_sum += (point_x_y[1] - mean_y_value)**2

        MSE = mean_error_sum / len(X_test)
        list_of_points_with_respective_predicted_y_coordinate = sorted(list_of_points_with_respective_predicted_y_coordinate, key=lambda tup: tup[0])
        return np.array(list_of_points_with_respective_predicted_y_coordinate), MSE
    
    def distances_and_mean_distance(self, point_in_X, k):
        train_points_with_distance = []
        for i in range(len(self.X_train)):

            # x_value of training data - x_value of test data
            distance = np.sqrt((self.X_train[i][0] - point_in_X[0])**2)

            # append training points with their distance to the considered x_value of the test data
            train_points_with_distance.append((self.X_train[i][0], self.X_train[i][1], distance))

        # sort the points in ascenting order based on the lowest distance (tup[2])
        train_points_with_distance = sorted(train_points_with_distance, key=lambda tup: tup[2])

        # mean value of y-coordinates of k nearest points
        mean = 0
        for i in range(k):

            # sum up k nearest y-coordinates
            mean += train_points_with_distance[i][1]

        # calculate mean value
        mean = mean / k
        return mean
