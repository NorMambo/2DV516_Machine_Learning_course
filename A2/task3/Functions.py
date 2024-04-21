import numpy as np

def normalize(X):
    
    # find the mean of every column
    mean = np.mean(X, axis=0)

    # find the standard dev of every column 
    std = np.std(X, axis=0)

    # ever x in X - the mean of the column / standard deviation of the column
    normalized_features = (X - mean)/std

    return normalized_features

def extend(data):
    ones = np.ones((data.shape[0], 1))
    extended_data = np.append(ones, data, axis=1)
    return extended_data

def sigmoid_function(X_norm):
    z = 1/(1 + np.exp(-X_norm))
    return z

def vectorized_gradient_descent(X, y, learning_rate, iterations):
    
    # Initialize betas with zeros
    theta = np.array(np.zeros(X.shape[1]))
    theta = np.reshape(theta, (X.shape[1], 1))

    cost_list = []

    for i in range(iterations):

        # Calculate predictions using current betas
        y_pred = sigmoid_function(np.dot(X, theta))

        # Calculate the error
        error = y_pred - y

        # Calculate the gradient of the cost function with respect to betas
        gradient = np.dot(X.T, error) / len(y)

        cost = logistic_cost_function(theta, X, y)
        # print(cost)
        cost_list.append(np.array(cost))

        # Update betas
        theta -= learning_rate * gradient

    cost_list = np.array(cost_list)
    return theta, cost_list

def logistic_cost_function(theta, x, y):

    y_pred = sigmoid_function(np.dot(x , theta))

    error = (y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred))

    cost = -1 / len(y) * sum(error)

    return cost