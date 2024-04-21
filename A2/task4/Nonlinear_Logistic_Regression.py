import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pylab
from sklearn.model_selection import train_test_split
import Functions as fs
import pandas as pd

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory, "microchips.csv")
train_data_complete = genfromtxt(path, delimiter=',')

X_train = np.array(train_data_complete[:, :2])


# ------------ create arrays of valid and failed microchips ------------
X_train_positive = []
X_train_negative = []
for row in train_data_complete:
    if row[2] == 1:
        X_train_positive.append(row)
    else:
        X_train_negative.append(row)
X_train_negative = np.array(X_train_negative)
X_train_positive = np.array(X_train_positive)

# ------------------ create y columns for train ----------------
y_train = np.array(train_data_complete[:, 2])
y_train = np.reshape(y_train, (-1, 1))


# 1) Plot the data in X and y using different symbols or colors for the two different classes.
# Notice also that X1 and X2 are already normalized. Hence, no need for normalization in this exercise.

plt.scatter(X_train_positive[:, 0], X_train_positive[:, 1], color="Green")
plt.scatter(X_train_negative[:, 0], X_train_negative[:, 1], color="Red")
plt.show()

# 2) Use gradient descent to find beta in the case of a quadratic model.

# CREATE COLUMNS TO EXTEND X1, X2 TO A QUADRATIC MODEL:
# -----------------------------------------------------
X1_square = np.square(X_train[:, 0])
X1_square = np.reshape(X1_square, (-1, 1))

X1_mul_X2 = []
for row in X_train:
    dot_p = np.dot(row[0], row[1])
    X1_mul_X2.append(np.array([dot_p]))
X1_mul_X2 = np.array(X1_mul_X2)

X2_square = np.square(X_train[:, 1])
X2_square = np.reshape(X2_square, (-1, 1))

X_train_for_nonlin_reg = np.c_[np.ones(X_train.shape[0]), X_train, X1_square, X1_mul_X2, X2_square]


# ----------------- gradient descent-------------------
learning_rate_1 = 5

iterations_1 = 10000


betas_1, cost_list_1 = fs.vectorized_gradient_descent(X_train_for_nonlin_reg, y_train, learning_rate_1, iterations_1)

print()
print("Learning rate (a):",learning_rate_1)
print()
print("Iterations:", iterations_1)
print()
print("TASK 2) Betas with quadratic model by gradient descent: ")
print(betas_1)
print()


X_train_extended = fs.map_feature(X_train[:, 0], X_train[:, 1], 2)

learning_rate_2 = 5

iterations_2 = 10000

betas_2, cost_list = fs.vectorized_gradient_descent(X_train_extended, y_train, learning_rate_2, iterations_2)
print("TASK 3) Betas by gradient descent (using Xe determined by map_feature function): ")
print(betas_2)

dot_p = np.dot(X_train_extended, betas_2)
probability = fs.sigmoid_function(dot_p)

fail_count = 0
for i, j in zip(probability, train_data_complete):
    if i >= 0.5 and j[2] == 0.0:
        fail_count += 1
    elif i < 0.5 and j[2] == 1.0:
        fail_count += 1

print(fail_count)

predicted_pos = []
predicted_neg = []
prob = []

for row in X_train_extended:
    dot_p = np.dot(row, betas_2)
    probability = fs.sigmoid_function(dot_p)
    prob.append(probability)
    if probability >= 0.5:
        predicted_pos.append(row)
    else:
        predicted_neg.append(row)

predicted_pos = np.array(predicted_pos)
predicted_neg = np.array(predicted_neg)
predicted_pos = np.delete(predicted_pos, 0, 1)
predicted_neg = np.delete(predicted_neg, 0, 1)


n_iterations = np.linspace(0, iterations_1 - 1, num=iterations_1)

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
fig = pylab.gcf()
fig.canvas.manager.set_window_title('Test')

# -------------------------------------------- PLOT 1 -------------------------------------------------

# LEFT PLOT
axs[0].plot(n_iterations, cost_list_1[:, 0])
axs[0].grid(True)
axs[0].set_xlabel("ITERATIONS")
axs[0].set_ylabel("COST")

# RIGHT PLOT
xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, betas_1, 2)
axs[1].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[1].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green", marker="." ,cmap=cmap_bold)
axs[1].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" , marker="." ,cmap=cmap_bold)
axs[1].set_title(f"Training Errors: {fail_count}")

plt.show()

# 4) Use mapFeatures to repeat (2) but with a polynomial of degree five (d = 5) model.

Xe = fs.map_feature(X_train[:, 0], X_train[:, 1], 5)

learning_rate_3 = 0.1

iterations_3 = 10000


betas3, cost_list = fs.vectorized_gradient_descent(Xe, y_train, learning_rate_2, iterations_2)

print("BETAS 4th EXERCISE (Xe determined by map_feature function): ")
print(betas3)

dot_p = np.dot(Xe, betas3)
probability = fs.sigmoid_function(dot_p)

fail_count = 0
for i, j in zip(probability, train_data_complete):
    if i >= 0.5 and j[2] == 0.0:
        fail_count += 1
    elif i < 0.5 and j[2] == 1.0:
        fail_count += 1
print("FAIL COUNT: ")
print(fail_count)
print()

predicted_pos = []
predicted_neg = []
prob = []

for row in Xe:
    dot_p = np.dot(row, betas3)
    probability = fs.sigmoid_function(dot_p)
    prob.append(probability)
    if probability >= 0.5:
        predicted_pos.append(row)
    else:
        predicted_neg.append(row)

predicted_pos = np.array(predicted_pos)
predicted_neg = np.array(predicted_neg)
predicted_pos = np.delete(predicted_pos, 0, 1)
predicted_neg = np.delete(predicted_neg, 0, 1)

# -------------------------------------------- PLOT 2 -------------------------------------------------

n_iterations = np.linspace(0, iterations_1 - 1, num=iterations_1)

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
fig = pylab.gcf()
fig.canvas.manager.set_window_title('Test')

# LEFT PLOT
axs[0].plot(n_iterations, cost_list_1[:, 0])
axs[0].grid(True)
axs[0].set_xlabel("ITERATIONS")
axs[0].set_ylabel("COST")

# RIGHT PLOT
xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, betas3, 5)
axs[1].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[1].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[1].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[1].set_title(f"Training Errors: {fail_count}")
plt.show()

# the question now is: is it correct to calculate the probabilities with 1, X1, X2 and then plot the boundaries with quadratic, cubic, ect model
# or should we find the probabilities with the extended version 1, X1, X2, X1ˆ2, X1*X2, X2ˆ2 and plot the boundaries like before?