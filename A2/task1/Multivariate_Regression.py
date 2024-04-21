import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import Functions as fs

directory = os.path.dirname(os.path.abspath(__file__))
csv_file = open(os.path.join(directory, "GPUbenchmark.csv"), "r")
csv_reader = csv.reader(csv_file, delimiter=",")

X = []
y = []

for row in csv_reader:
    details = np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
    benchmark_speed = np.array([float(row[6])])
    X.append(details)
    y.append(benchmark_speed)

X = np.array(X)
y = np.array(y)

# 1) Start by normalizing X using Xn = (X - u)/o.

X_norm = fs.normalize(X)
X_norm_extended = fs.extend(X_norm)

print("PART 1)")
print()
print("Normalized dataset: ")
print(X_norm)
print()

# 2. Multivariate datasets are hard to visualize. However, to get a basic understanding it
#    might be a good idea to produce a plot Xi vs y for each one of the features.
#    Use subplot(2,3,i) to fit all six plots into a single figure. Make sure that each nomalized Xi is centralized around zero.

fig, axs = plt.subplots(2, 3, figsize=(15, 7))

axs[0][0].scatter(X_norm[:, 0], y, s=5, edgecolors='k')
axs[0][1].scatter(X_norm[:, 1], y, s=5, edgecolors='k')
axs[0][2].scatter(X_norm[:, 2], y, s=5, edgecolors='k')
axs[1][0].scatter(X_norm[:, 3], y, s=5, edgecolors='k')
axs[1][1].scatter(X_norm[:, 4], y, s=5, edgecolors='k')
axs[1][2].scatter(X_norm[:, 5], y, s=5, edgecolors='k')

plt.show()

# 3. Compute 􏰁 using the normal equation 􏰁 = (X.T * X)ˆ-1 * X.T * y, where X is the extended normalized matrix [1; X1; : : : ; X6].
#    What is the predicted benchmark result for a graphic card with the following (non-normalized) feature values?
#    2432; 1607; 1683; 8; 8; 256
#    The actual benchmark result is 114.

print("PART 3)")
print()

# add the new row to the data set and find its normalized value
X1 = X
new_row = [float(2432), float(1607), float(1683), float(8), float(8), float(256)]
X1 = np.vstack([X1, new_row])

X2 = fs.extend(X1)

betas = fs.linear_regression_normal_equation(X, y)

print("Beta values by Normal Equation: ")
print(betas)

# prdict y for normalized values of 2432; 1607; 1683; 8; 8; 256
y_predicted = X2[18].dot(betas)

print()
print("Predicted benchmark result by Normal Equation: ")
print(y_predicted)
print()
print("Difference between actual and predicted benchmark result: ")
print(114 - y_predicted)
print()

# 4) What is the cost J(betas) when using the betas computed by the normal equation above?

print("PART 4)")
print()
Xe = fs.extend(X)
cost = fs.cost_function(Xe, y, betas, len(y))

print("Cost J(beta) for beta computed by Normal Equation: ")
print(cost)
print()

# 5) Gradient descent (a)

print("PART 5a)")
print()

learning_rate = 0.4
iterations = 1000

# betas2 = fs.unvectorized_gradient_descent(X_norm_for_gradient_descent, y, learning_rate, iterations)
betas2 = fs.vectorized_gradient_descent(X_norm_extended, y, learning_rate, iterations)

print("Beta values by Gradient Descent: ")
print(betas2)
print()

cost2 = fs.cost_function(X_norm_extended, y, betas2, len(X))
print("Cost J(beta) for beta computed by Gradient Descent: ")
print(cost2)
print()
print("Cost divided by 100 calculated by Normal Equation: ")
print(cost/100)
print()
print("Cost divided by 100 calculated by Gradient Descent: ")
print(cost2/100)
print()
print(f"By using learning rate: {learning_rate} and iterations: {iterations} the difference stays within 1% of the cost calculated by Normal Equation")
print()

# 5) Gradient descent (b)

# find the y prediction by dot product (equivalent of: Y = b0 + x1*b1 + x2*b2 +... + Xn*bn)
X_norm_with_added_val = fs.normalize(X1)
X_norm_extended_with_added_val = fs.extend(X_norm_with_added_val)

y_predicted_2 = X_norm_extended_with_added_val[18].dot(betas2)



print("PART 5b)")
print()
print("Predicted benchmark result by Gradient Descent: ")
print(y_predicted_2)
print()