import numpy as np
from numpy.random import default_rng
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import Functions as fs


directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory, "bm.csv")

# 10.000 data points
data = np.genfromtxt(path, delimiter=',')

# Shuffle the data
np.random.seed(1)
np.random.shuffle(data)

# Split data 90/10
(train_set, test_set) = train_test_split(data, test_size=0.1)

# divide test_set into X and y
X_test = test_set[:, :2]
y_test = test_set[:, 2]
y_test = y_test.reshape((y_test.shape[0], 1))

rng = default_rng()

# Create 100 subsets to train the decision trees
subsets_100_X_train = []
subsets_100_y_train = []

# randomize 100 train sets of 5000 data points each
for i in range(100):

    subset = []

    for i in range(5000):
        subset.append(train_set[rng.integers(low=0, high=9000)])

    subset = np.array(subset)

    # Split the created subset into X an y
    X = subset[:, :2]
    y = subset[:, 2]
    y = y.reshape((y.shape[0], 1))

    # Append every created subset to a subset list
    subsets_100_X_train.append(X)
    subsets_100_y_train.append(y)

# Create and fit 100 DecisionTreeClassiefiers
Decision_trees_100 = []
for i in range(100):
    tree = DecisionTreeClassifier()
    tree.fit(subsets_100_X_train[i], subsets_100_y_train[i])
    Decision_trees_100.append(tree)

# -------------------------------------------------------------------------------
# Create an accuracy list to get the mean in part B
accuracy_list = []

# Create a prediction list to perform majority vote in part A
prediction_list = []

# Predict and append accuracies to accuracy list
for tree in Decision_trees_100:
    correct = 0
    y_pred = tree.predict(X_test)
    y_pred = y_pred.reshape((y_pred.shape[0], 1))

    prediction_list.append(y_pred)

    # Find correctly predicted for every single predicted set (100 in total)
    for i, j in zip(y_pred, y_test):
        if i == j:
            correct += 1

    accuracy = correct/len(y_test)
    accuracy_list.append(accuracy)

# -------------------------------------------------------------------------------
# Create a list of values to be selected by majority vote
majority_vote_y_pred = []

# Majority vote
for i in range(1000):
    one = 0
    zero = 0
    for j in range(len(prediction_list)):
        if prediction_list[j][i] == 1:
            one += 1
        else:
            zero += 1
    if one > zero:
        majority_vote_y_pred.append(np.array(1))
    else:
        majority_vote_y_pred.append(np.array(0))
    
majority_vote_y_pred = np.array(majority_vote_y_pred)

# Find correctly predicted by majority vote
correct = 0
for i, j in zip(majority_vote_y_pred, y_test):
    if i == j:
        correct += 1

# Find accuracy in y_pred selected by majority vote
accuracy = correct/len(y_test)
accuracy_percentage = round(accuracy * 100, 2)
gener_error = round((1 - accuracy) * 100, 2)

print("\n(A)")
print(f"Accuracy for majority vote in forest is: {accuracy_percentage} %")
print(f"Generalizaion error for majority vote in forest is: {gener_error} %")

# Find mean accuracy in y_pred selected by majority vote
mean_accuracy = np.mean(accuracy_list)
mean_accuracy_percentage = round(mean_accuracy * 100, 2)
generalization_error = round((1 - mean_accuracy) * 100, 2)

print("\n(B)")
print(f"Mean accuracy for 100 trees is: {mean_accuracy_percentage} %")
print(f"Generalization error for 100 trees is: {generalization_error} %\n")


fs.plot_boundary(Decision_trees_100, X_test)