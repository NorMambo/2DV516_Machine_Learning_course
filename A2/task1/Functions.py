import numpy as np

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    normalized_features = (X - mean)/std
    return normalized_features

def linear_regression_normal_equation(X, y):
    ones = np.ones((X.shape[0], 1))
    X = np.append(ones, X, axis=1)
    # W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # W contains beta 1, beta 2, beta 3
    return betas

def cost_function(X, y, beta_values, n):
    
    # add column of 1s to X
    # ones = np.ones((X.shape[0], 1))
    # X = np.append(ones, X, axis=1)

    # compute cost
    cost = np.sum(np.square(np.dot(X, beta_values) - y))/n

    return cost

def vectorized_gradient_descent(X, y, learning_rate, iterations):
    
    # Initialize betas with random zeros
    theta = np.array(np.zeros(X.shape[1]))
    theta = np.reshape(theta, (X.shape[1], 1))
    print(theta)

    # gradient descent
    for i in range(iterations):
        # Calculate predictions using current betas
        y_pred = np.dot(X, theta)

        # Calculate the error
        error = y_pred - y

        # Calculate the gradient of the cost function with respect to betas
        gradient = np.dot(X.T, error) / len(y)
        cost = cost_function(X, y, theta, len(y))
        # print(cost)

        # Update betas
        theta -= learning_rate * gradient

    return theta



# this function is part of "unvectorized gradient descent"
# parameters: normalized dataset, sample y values, set of zeros equal to the number of columns
def _cost(data, actual_y, params):
    total_cost = 0
    for i in range(data.shape[0]):
        total_cost += (1/data.shape[0]) * ((data[i] * params).sum() - actual_y[i])**2
    return total_cost

# calculate gradient descent
# parameters: pass a normalized dataset, the set of sample y values, a set of zeros equal to the number of columns, an arbitrary learning rate, a number of iterations
# NOTE: uncomment the print statement if you want to see the descent!
def unvectorized_gradient_descent(data, actual_y, params, learning_rate, iterations):
    # list_of_costs = []
    for i in range(iterations):
        slopes = np.zeros(data.shape[1])
        for j in range(data.shape[0]):
            for k in range(data.shape[1]):
                slopes[k] += (1/data.shape[0]) * ((data[j] * params).sum() - actual_y[j]) * data[j][k]
        params = params - learning_rate * slopes
        # print(_cost(data, actual_y, params))
        # list_of_costs.append(_cost(data, actual_y, params))

    betas = []
    for i in params:
        a = np.array([i])
        betas.append(a)
    betas = np.array(betas)    
                    
    return betas

def extend(data):
    ones = np.ones((data.shape[0], 1))
    extended_data = np.append(ones, data, axis=1)

    return extended_data
