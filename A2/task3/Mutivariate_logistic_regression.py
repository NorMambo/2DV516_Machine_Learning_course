import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import Functions as fs

# 1) Read data and shuffle the rows in the raw data matrix:
directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory, "banknote_authentication.csv")

data = genfromtxt(path, delimiter=',')

Data_train, Data_test = train_test_split(data, train_size=0.05, random_state=None ,shuffle=True) # train_size=0.5, random_state=1234
print(Data_train.shape)
print(Data_test.shape)

# 2) Divide the data into suitable sized train and test sets.

X_train = np.array(Data_train[:, :4])
X_test = np.array(Data_test[:, :4])

y_train = np.array(Data_train[:, 4])
y_train = np.reshape(y_train, (-1, 1))

y_test = np.array(Data_test[:, 4])
y_test = np.reshape(y_test, (-1, 1)) 

# 3) Normalize the training data and train a linear logistic regression model using gradient descent.
# Print the hyperparameters a and N(iter) and plot the cost function J(beta) as a function over iterations.

X_train_norm = fs.normalize(X_train)
X_train_norm_extended = fs.extend(X_train_norm)

learning_rate = 0.05
iterations = 600

betas, cost_list = fs.vectorized_gradient_descent(X_train_norm_extended, y_train, learning_rate, iterations)

print()
print("Learning rate (a):",learning_rate)
print()
print("Iterations:", iterations)
print()
print("Betas")
print(betas)
print()

n_iterations = np.linspace(0, iterations - 1, num=iterations)

plt.plot( n_iterations, cost_list[:, 0])
plt.grid(True)
plt.xlabel("ITERATIONS")
plt.ylabel("COST")
plt.title("COST vs ITERATIONS")
plt.show()

# 4) What is the training error (number of non-correct classifications in the training data)
# and the training accuracy (percentage of correct classifications) for your model?

train_fail_count = 0
for i in range(0, len(X_train_norm_extended)):

    dot_p = np.dot(X_train_norm_extended[i], betas)
    probability = fs.sigmoid_function(dot_p)
    if probability >= 0.5:
        y_pred = 1
    else:
        y_pred = 0
    if y_pred != y_train[i]:
        train_fail_count+=1

print("Training error: ")
print(train_fail_count)
print()

print("Training accuracy: ")
training_accuracy = (len(X_train_norm) - train_fail_count)/len(X_train_norm)
print(training_accuracy)
print()

# 5) What is the number of test error and the test accuracy for your model?

X_test_norm = fs.normalize(X_test)
X_test_norm_extended = fs.extend(X_test_norm)

test_fail_count = 0
for i in range(0, len(X_test_norm_extended)):

    dot_p = np.dot(X_test_norm_extended[i], betas)
    probability = fs.sigmoid_function(dot_p)
    if probability >= 0.5:
        y_pred = 1
    else:
        y_pred = 0
    if y_pred != y_test[i]:
        test_fail_count+=1

print("Testing error: ")
print(test_fail_count)
print()

print("Testing accuracy: ")
testing_accuracy = (len(X_test_norm) - test_fail_count)/len(X_test_norm)
print(testing_accuracy)
print()

# 6) Repeated runs will (due to the shuffling) give different results.
# Are they qualitatively the same? Do they depend on how many observations you put aside for testing? 
# Is the difference between training and testing expected?

# SEE README