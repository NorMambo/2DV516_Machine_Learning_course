import os
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# ------------------------------------ Read data --------------------------------------
directory = os.path.dirname(os.path.abspath(__file__))

# microchips file contains 2 values followed by 1 (OK) or 0 (FAILED)
csv_file = open(os.path.join(directory, "data", "polynomial200.csv"), "r")

# csv.reader returns an iterable: csv_reader
csv_reader = csv.reader(csv_file, delimiter=',')

# create list of x values
XY_values = []

# fill the 2 lists with the values provided in the .csv files and USING NUMPY
for i in csv_reader:
    a = np.array([float(i[0]), float(i[1])])
    XY_values.append(a)

XY_values = np.array(XY_values)

# -------------------- (EX 2.1) Divide dataset into 2 equal parts ----------------------

# use train_test_split to split data in half
xy_train, xy_test = train_test_split(XY_values, test_size=0.5, random_state=1234, shuffle=False)

# ------------------------ (EX 2.2) Plot train and test data ---------------------------

fig, (ax) = plt.subplots(1, 2, figsize=(10, 7))

fig.canvas.manager.set_window_title("TRAINING DATA --- TEST DATA")

ax[0].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
ax[0].set_aspect('equal', 'box')
ax[0].set_title("Training data")

ax[1].scatter(xy_test[:, 0], xy_test[:, 1], edgecolors='k')
ax[1].set_aspect('equal', 'box')
ax[1].set_title("Test data")

plt.show()

# ---------------- (EX 2.3) show training regression and training error -----------------

from Polynomial_KNN_class import KNN_regression

def KNN_reg_with_k(k):
    clf = KNN_regression(k)
    clf.fit(xy_train)
    # passing the train values as parameter
    arr, MSE = clf.predict(xy_train)
    return arr, round(MSE, 2)
fig, axs = plt.subplots(2, 3, figsize=(15, 7))

fig.canvas.manager.set_window_title("REGRESSION OF TRAINING DATA + TRAINING ERROR")

arr, MSE = KNN_reg_with_k(1)
axs[0][0].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[0][0].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[0][0].set_title(f"K = 1, MSE = {MSE}")

arr, MSE = KNN_reg_with_k(3)
axs[0][1].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[0][1].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[0][1].set_title(f"K = 3, MSE = {MSE}")

arr, MSE = KNN_reg_with_k(5)
axs[0][2].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[0][2].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[0][2].set_title(f"K = 5, MSE = {MSE}")

arr, MSE = KNN_reg_with_k(7)
axs[1][0].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[1][0].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[1][0].set_title(f"K = 7, MSE = {MSE}")

arr, MSE = KNN_reg_with_k(9)
axs[1][1].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[1][1].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[1][1].set_title(f"K = 9, MSE = {MSE}")

arr, MSE = KNN_reg_with_k(11)
axs[1][2].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[1][2].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[1][2].set_title(f"K = 11, MSE = {MSE}")
plt.show()

# ------------------ (EX 2.4) show test regression and test error ----------------------

# NOTE: the computation and presentation is done exactly as in ex 2.3 but with test
# values instead of the training values

def KNN_reg_with_k(k):
    clf = KNN_regression(k)
    clf.fit(xy_train)
    # passing the test values as parameter
    arr, MSE = clf.predict(xy_test)
    return arr, round(MSE, 2)

fig, axs = plt.subplots(2, 3, figsize=(15, 7))

fig.canvas.manager.set_window_title("REGRESSION OF TEST DATA + TRAINING ERROR")

arr, MSE = KNN_reg_with_k(1)
axs[0][0].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[0][0].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[0][0].set_title(f"K = 1, MSE = {MSE}")

arr, MSE = KNN_reg_with_k(3)
axs[0][1].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[0][1].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[0][1].set_title(f"K = 3, MSE = {MSE}")

arr, MSE = KNN_reg_with_k(5)
axs[0][2].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[0][2].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[0][2].set_title(f"K = 5, MSE = {MSE}")

arr, MSE = KNN_reg_with_k(7)
axs[1][0].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[1][0].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[1][0].set_title(f"K = 7, MSE = {MSE}")

arr, MSE = KNN_reg_with_k(9)
axs[1][1].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[1][1].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[1][1].set_title(f"K = 9, MSE = {MSE}")

arr, MSE = KNN_reg_with_k(11)
axs[1][2].plot(arr[:, 0], arr[:, 1], c='Blue')
axs[1][2].scatter(xy_train[:, 0], xy_train[:, 1], edgecolors='k')
axs[1][2].set_title(f"K = 11, MSE = {MSE}")
plt.show()

# --------------------- (EX 2.5) MOTIVATE BEST REGRESSION ------------------------

# I think that since the training and the test data stem from the same set of values
# and are actually very similar, a low value of k such as 3 better represents the 
# regression. Many points of the training data are located where the test data should
# approximately be located, so I think that k = 3 is a good fit, since it captures
# the nearest 3 neighbors. If the value increases, the curve flattens and it starts
# becoming less precise.