import numpy as np
from matplotlib.colors import ListedColormap

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

def find_prediction_with_betas(X, betas):

    count_passed = 0
    count_failed = 0

    for row in X:
        probability = np.dot(row, betas)
        probability = sigmoid_function(probability)
        if probability >= 0.5:
            count_passed += 1
        else:
            count_failed += 1

    return count_passed, count_failed

def map_feature(X1,X2,degree): # Pyton

    one = np.ones([len(X1),1])
    Xe = np.c_[one,X1,X2] # Start with [1,X1,X2]

    for i in range(2,degree+1):
        
        for j in range(0,i+1):
            Xnew = np.multiply(np.power(X1, (i-j)), np.power(X2, j)) # type (N)
            Xnew = Xnew.reshape(-1,1) # type (N,1) required by append
            Xe = np.append(Xe,Xnew,1) # axis = 1 ==> append column
    return Xe

def create_decision_boundary(X, betas, degree):
    h = .01 # step size in the mesh

    x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) # Mesh Grid

    x1,x2 = xx.ravel(), yy.ravel() # Turn to two Nx1 arrays

    XXe = map_feature(x1,x2,degree) # Extend matrix for degree 2

    p = sigmoid_function( np.dot(XXe, betas) ) # classify mesh ==> probabilities XXe

    classes = p>0.5 # round off probabilities
    clz_mesh = classes.reshape(xx.shape) # return to mesh format
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]) # mesh plot
    cmap_bold = ListedColormap(["FF0000", "#00FF00", "#0000FF"]) # colors
    return xx, yy, clz_mesh, cmap_light, cmap_bold
