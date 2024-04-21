import os
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import Functions as fs

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory, "cars-mpg.csv")
data = genfromtxt(path, delimiter=',', skip_header=True)

# 1) Start by splitting the data 4 : 1 as training and validation randomly (for grading purposes, please use np.random.seed(1)).

np.random.seed(1)
np.random.shuffle(data)

X_train, X_val, y_train, y_val = train_test_split(data[:, 1:], data[:, 0], test_size=0.25, random_state=None, shuffle=False)

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

# 2) Implement the forward selection algorithm as discussed in Lecture 6 (see lecture notes for details).
#    In the loop use the training MSE to find the best model in each iteration. 
#    The algorithm should produce p + 1 models M0; : : : ; Mp, where Mi is the best model using i features. 
#    In terms of output, an alternative could be to let the algorithm produce a p-dimensional vector where 
#    its first entry is the feature in M1, its second entry is the new feature in M2 etc.

columns_and_MSE_train_set = fs.forward_selection(X_train, y_train)
print()
print("TASK 2:")
print("SEQUENCE OF FEATURES AND MSE AFTER ALWAYS ADDING 1 MORE TO THE SET: ")
print(columns_and_MSE_train_set)

indexes = []
for i in columns_and_MSE_train_set:
    indexes.append(i[0][0])

print()
print("INDEXES OF BEST MODELS")
print(indexes)

# 3. Apply your forward selection on the GPUbenchmark.csv. Use the validation set to find the best model
#    among all Mi, i = 1;:::;6. Which is the best model? Which is the most important feature, i.e. selected first?
#    (Sequence is: 3, 35, 356, 3561, 35614, 356142, 3561420)

val_feature_list = []
train_feature_list = []

for i in range(0, X_val.shape[1]):
    column_of_features = X_val[:, i]
    val_feature_list.append(column_of_features)

for i in range(0, X_train.shape[1]):
    column_of_features = X_train[:, i]
    train_feature_list.append(column_of_features)

list_of_val_models_for_evaluation = []
list_of_train_models_for_evaluation = []

ones = np.ones((X_val.shape[0], 1))

for i in range(0, len(indexes) + 1):
    Xe = ones
    for j in range(0, i):
        Xe = np.c_[Xe, val_feature_list[indexes[j]]]
    list_of_val_models_for_evaluation.append(Xe)
# for i in list_of_val_models_for_evaluation:
#     print(i[:3, :])


ones = np.ones((X_train.shape[0], 1))

for i in range(0, len(indexes) + 1):
    Xe = ones
    for j in range(0, i):
        Xe = np.c_[Xe, train_feature_list[indexes[j]]]
    list_of_train_models_for_evaluation.append(Xe)

# list_of_models_for_evaluation IS CORRECT

list_of_MSE = []

n = y_val.shape[0]

for train_m, val_m, i in zip(list_of_train_models_for_evaluation, list_of_val_models_for_evaluation, range(1, 9)):
    lin_reg = LinearRegression()
    lin_reg.fit(train_m, y_train)
    y_pred = lin_reg.predict(val_m)
    MSE = np.sum(np.square(np.subtract(y_val, y_pred)))/n
    list_of_MSE.append((f"With {i} column(s), MSE = ", round(MSE, 4), i))

print()
print("MSE's: ")
for i in list_of_MSE:
    print(i)

min = 10000
best_model = ()
nr_of_columns = 0
for i in list_of_MSE:
    if i[1] < min:
        min = i[1]
        best_model = i
        nr_of_columns = i[2]

print()
print("BEST MODEL (including the column of 1's): ")
print(best_model)
print()

important_indexes = []
print("Sequence of included feature columns ordered by importance (after the column of 1's): ")
for i in range(0, nr_of_columns - 1):
    important_indexes.append(indexes[i])

print(important_indexes)
print()
print(f"The most important feature column (selected first) is column {important_indexes[0]}")
print()








