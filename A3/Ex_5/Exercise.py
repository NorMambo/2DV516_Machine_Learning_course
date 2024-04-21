import os
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

directory = os.path.dirname(os.path.abspath(__file__))
path1 = os.path.join(directory, "fashion-mnist_train.csv")
path2 = os.path.join(directory, "fashion-mnist_test.csv")

# train data = 60.000 rows / 785 columns
train_data = np.genfromtxt(path1, delimiter=",", skip_header=1)
X_train = train_data[:, 1:]
y_train = train_data[:, 0]

# test data = 10.000 rows / 785 columns
test_data = np.genfromtxt(path2, delimiter=",", skip_header=1)
X_test = test_data[:, 1:]
y_test = test_data[:, 0]

# 1) PLOT DATA

X_train_for_plot = np.reshape(X_train, (X_train.shape[0], 28, 28))
rng = default_rng()

for i in range(16):  
    plt.subplot(4, 4, i + 1)
    rand_index = rng.integers(low=0, high=60000)
    plt.imshow(X_train_for_plot[rand_index], cmap=plt.get_cmap('gray'))
    plt.tick_params(axis='x', labelbottom=False)
    plt.tick_params(axis='y', labelleft=False)
    plt.ylabel(f"{int(y_train[rand_index])}")
plt.show()

# 2) Train a multilayer perceptron to achieve as good accuracy as you can.
#    There are numerous hyperpa- rameters that we discussed in class which you can tweak,
#    for instance: learning rate, number of and size of hidden layers, activation function
#    and regularization (e.g. Ridge (known here as L2), and early stopping).
#    You should make a structured search for the best hyperparameters that you can find.

# Normalize data
X_train = X_train/255
X_test = X_test/255

# Create set for GridSearchCV
X_for_gridsearch = X_train[0:2000, :]
y_for_gridsearch = y_train[0:2000]

# UNCOMMENT DOWN HERE IF YOU WANT TO RUN GRIDSEARCH WITH THE FOLLOWING PARAMS (takes hours to run...)
# ---------------------------------------------------------------------------------------------------
# clf = MLPClassifier()

# parameters={
# 'learning_rate': ["constant", "invscaling", "adaptive"],
# 'hidden_layer_sizes': [(10,), (50,), (100,)],
# 'max_iter': [1000],
# 'alpha': [0.001, 0.01, 0.1],
# 'activation': ["identity", "logistic", "relu", "tanh"]
# }

# grid = GridSearchCV(estimator=clf,param_grid=parameters,n_jobs=-1,verbose=2,cv=10)
# grid.fit(X_for_gridsearch, y_for_gridsearch)
# print(grid.best_params_)
# ---------------------------------------------------------------------------------------------------

# Best Hyperparameters found with GridSearchCV (hidden layer sizes were 10, 20, 30) the first time I ran it
best_param = {'activation': 'identity', 'alpha': 0.001, 'hidden_layer_sizes': (30,), 'learning_rate': 'adaptive', 'max_iter': 1000}

# The 2nd time, Grid search ran for almost 2.5 hours to find the following params (accuracy around 85% with them)
best_param = {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100,), 'learning_rate': 'invscaling', 'max_iter': 1000}

# Create a MLPClassifier with the best hyperparameters found with GridSearchCV
clf = MLPClassifier(activation=best_param['activation'], alpha=best_param['alpha'], hidden_layer_sizes=best_param['hidden_layer_sizes'], learning_rate=best_param['learning_rate'], max_iter=best_param['max_iter'])

# Take smaller subset of normalized X_train and y_train
X_train = X_train[0:10000, :]
y_train = y_train[0:10000]

# Take smaller subset of normalized X_test and y_test
X_test = X_test[0:2000, :]
y_test = y_test[0:2000]

# Fit with subset of 10.000 train values
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Find correct predictions
correct = 0
for i,j in zip(y_pred, y_test):
    if i == j:
        correct+=1

# Find accuracy
accuracy = correct/len(y_test) * 100
print(f"\nPrediction accuracy with MLPClassifier and best hyperparameters: {accuracy} %")


# 3) Plot the confusion matrix. Which are the easy/hard categories to classify? 
#    Are there any particular classes that often gets mixed together?

print("\nConfusion Matrix:")
print(confusion_matrix(y_pred, y_test))



