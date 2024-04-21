import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import Functions as fs

directory = os.path.dirname(os.path.abspath(__file__))
csv_file = open(os.path.join(directory, "secret_polynomial.csv"), "r")
csv_reader = csv.reader(csv_file)

xy = []

for row in csv_reader:
    xy_val = np.array([float(row[0]), float(row[1])])
    xy.append(xy_val)

xy = np.array(xy)
xy_train_set, xy_test_set = train_test_split(xy, test_size=0.5, random_state=None, shuffle=True)
count = 0

def polynomial_regression(xy_train, xy_test, degree):

    X_train = np.reshape(xy_train[:, 0], (-1, 1))
    X_test = np.reshape(xy_test[:, 0], (-1, 1))

    y_train = np.reshape(xy_train[:, 1], (-1, 1))
    y_test = np.reshape(xy_test[:, 1], (-1, 1))

    Xe_train = fs.extend(X_train, degree)
    Xe_test = fs.extend(X_test, degree)

    b_train = fs.normal_equation(Xe_train, y_train)
    b_test = fs.normal_equation(Xe_test, y_test)

    y_predicted_train = np.dot(Xe_train, b_train)
    y_predicted_test = np.dot(Xe_test, b_test)

    MSE_train = np.sum(np.square(np.subtract(y_train, y_predicted_train)))/y_train.shape[0]
    MSE_test = np.sum(np.square(np.subtract(y_test, y_predicted_test)))/y_test.shape[0]

    return xy_train, X_train, y_predicted_train, round(MSE_train), round(MSE_test), degree


fig, axs = plt.subplots(2, 3, figsize=(15, 7))

xy_train, x_train, y_predicted, MSE_train, MSE_test, degree = polynomial_regression(xy_train_set, xy_test_set, 1)
axs[0][0].scatter(xy_train[:, 0], xy_train[:, 1], s=5, edgecolors='k')
axs[0][0].scatter(x_train, y_predicted, s=5)
axs[0][0].set_title(f"D: {degree} MSE train: {MSE_train} MSE test: {MSE_test}")

xy_train, x_train, y_predicted, MSE_train, MSE_test, degree = polynomial_regression(xy_train_set, xy_test_set, 2)
axs[0][1].scatter(xy_train[:, 0], xy_train[:, 1], s=5, edgecolors='k')
axs[0][1].scatter(x_train, y_predicted, s=5)
axs[0][1].set_title(f"D: {degree} MSE train: {MSE_train} MSE test: {MSE_test}")

xy_train, x_train, y_predicted, MSE_train, MSE_test, degree = polynomial_regression(xy_train_set, xy_test_set, 3)
axs[0][2].scatter(xy_train[:, 0], xy_train[:, 1], s=5, edgecolors='k')
axs[0][2].scatter(x_train, y_predicted, s=5)
axs[0][2].set_title(f"D: {degree} MSE train: {MSE_train} MSE test: {MSE_test}")

xy_train, x_train, y_predicted, MSE_train, MSE_test, degree = polynomial_regression(xy_train_set, xy_test_set, 4)
axs[1][0].scatter(xy_train[:, 0], xy_train[:, 1], s=5, edgecolors='k')
axs[1][0].scatter(x_train, y_predicted, s=5)
axs[1][0].set_title(f"D: {degree} MSE train: {MSE_train} MSE test: {MSE_test}")

xy_train, x_train, y_predicted, MSE_train, MSE_test, degree = polynomial_regression(xy_train_set, xy_test_set, 5)
axs[1][1].scatter(xy_train[:, 0], xy_train[:, 1], s=5, edgecolors='k')
axs[1][1].scatter(x_train, y_predicted, s=5)
axs[1][1].set_title(f"D: {degree} MSE train: {MSE_train} MSE test: {MSE_test}")

xy_train, x_train, y_predicted, MSE_train, MSE_test, degree = polynomial_regression(xy_train_set, xy_test_set, 6)
axs[1][2].scatter(xy_train[:, 0], xy_train[:, 1], s=5, edgecolors='k')
axs[1][2].scatter(x_train, y_predicted, s=5)
axs[1][2].set_title(f"D: {degree} MSE train: {MSE_train} MSE test: {MSE_test}")

plt.show()
