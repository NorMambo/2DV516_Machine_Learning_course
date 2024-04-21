import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import Functions as fs

directory = os.path.dirname(os.path.abspath(__file__))
path1 = os.path.join(directory, "dist.csv")
path2 = os.path.join(directory, "dist_val.csv")
data = np.genfromtxt(path1, delimiter=';')

# Divide data into X and y
X, y = data[:, :2], data[:, 2]

# NOTE: I did not use the validation set to find the hyperparameters, as the exercise says that the dist_val.csv set CAN be used for the validation of hyperparameters.

# 1) Tune the necessary hyperparameters by for instance grid search. In this exercise we are concerned with the hyperparameters given in Table 2.
#    Every hyperparameter should be tested for at least 3 values but you are free to add more testings.
#    There is a desginated validation set that can be used for the validation of the hyperparameters dist_val.csv.

# Selected values to perform GridSearchCV
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [1, 10, 100, 1000], 'degree': [1, 2, 3, 4, 5, 6], 'kernel': ['poly'], 'coef0': [0, 1, 2, 3, 4]}
]

# UNCOMMENT BELOW TO RUN GRIDSEARCH
# --------------------------------------------------------------------------------------------------
# clf_1 = SVC(kernel="linear")

# clf_2 = SVC(kernel="rbf")

# clf_3 = SVC(kernel="poly")

# grid1 = GridSearchCV(estimator=clf_1, param_grid=param_grid[0], refit=True, verbose=3, n_jobs=-1)
# grid2 = GridSearchCV(estimator=clf_2, param_grid=param_grid[1], refit=True, verbose=3, n_jobs=-1)
# grid3 = GridSearchCV(estimator=clf_3, param_grid=param_grid[2], refit=True, verbose=3, n_jobs=-1)

# grid1.fit(X, y)
# print()
# print("BEST PARAMS LINEAR")
# print(grid1.best_params_)
# print()

# grid2.fit(X, y)
# print()
# print("BEST PARAMS RBF")
# print(grid2.best_params_)
# print()

# grid3.fit(X, y)
# print()
# print("BEST PARAMS POLYNOMIAL")
# print(grid3.best_params_)
# print()
# --------------------------------------------------------------------------------------------------

# Best parameters found with GridSearchCV
param_rbf = {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
param_linear = {'C': 1, 'kernel': 'linear'}
param_poly = {'C': 1, 'coef0': 2, 'degree': 2, 'kernel': 'poly'}

# Fit the classifiers with parameter dictionary values
clf_linear = SVC(kernel=param_linear['kernel'], C=param_linear['C'])
clf_linear.fit(X, y)

# Fit the classifiers with parameter dictionary values
clf_rbf = SVC(kernel=param_rbf['kernel'], C=param_rbf['C'], gamma=param_rbf['gamma'])
clf_rbf.fit(X, y)

# Fit the classifiers with parameter dictionary values
clf_polynomial = SVC(kernel=param_poly['kernel'], C=param_poly['C'], degree=param_poly['degree'], coef0=param_poly['coef0'])
clf_polynomial.fit(X, y)


# 2) For each kernel, produce a plot of the decision boundary for the best models
#    together with the data. If you want you can also include the true decision boundary as a comparison.

# Plot boundaries
fs.plot_boundary(clf_linear, X, y)
fs.plot_boundary(clf_rbf, X, y)
fs.plot_boundary(clf_polynomial, X, y)

