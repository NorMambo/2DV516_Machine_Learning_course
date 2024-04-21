import numpy as np
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import Functions as fs

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory, "bm.csv")
data = np.genfromtxt(path, delimiter=',') # 10.000 data points

X, y = data[:, :2], data[:, 2]


# 1) Create a dataset which consists of a random sample of 5,000 datapoints in bm.csv.
#    To be able to compare with the subsequent results you can use the following code.

# Create dataset by adapting the code from the exercise sheet to this code
n_s = 5000
np.random.seed(7)
r = np.random.permutation(len(y))
X, y = X[r, :], y[r]
y = y.reshape(-1, 1)

X_s, y_s = X[:n_s, :], y[:n_s]
y_s = y_s.reshape(y_s.shape[0], 1)

# 2) Use sklearn to create and train a support vector machine using a Gaussian kernel 
#    and compute its training error (γ = .5 and C = 20 should yield a training error 
#    of .0102, however note that these hyperparams are not optimized and the results may be improved).

# Create classifier with given parameters
clf = SVC(gamma=0.5, C=20, kernel="rbf")

# Fit classifier
clf.fit(X_s, y_s)

# Predict labels
y_pred = clf.predict(X_s)

# find errors
fail_count = 0
for i, j in zip(y_s, y_pred):
    if i != j:
        fail_count += 1

# Find training error
training_error = fail_count/len(y_s)
print("\nTask 2) Training error: ")
print(training_error)
print()

# Get support vector indexes
support_vectors_indexes = clf.support_

support_vector_list = []
count = 0

# Use support vector indexes to find support vectors and add to a list
for i in support_vectors_indexes:
    support_vector_list.append(X_s[i])

# Tranform support vector list into a np array
support_vectors = np.array(support_vector_list)


# 3) Plot the decision boundary, the data and the support vectors in two plots, 
#    c.f. Figure 1. The indices of support vectors are obtained by clf.support_, where clf is your trained model.

fs.first_plot(clf, X_s, support_vectors)
fs.second_plot(clf, X_s, y_s)
