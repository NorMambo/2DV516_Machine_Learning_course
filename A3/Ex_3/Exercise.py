import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from keras.datasets import mnist
import Functions as fs


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# PLOT DATA
for i in range(9):  
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
plt.show()

# reshape to vector of 784
X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)

# Normalizing
X_train = X_train/255
X_test = X_test/255

# Diminish the train and test data size
X_train = X_train[0:10000, :]
y_train = y_train[0:10000]

X_test = X_test[0:2000, :]
y_test = y_test[0:2000]

# PERFORM GRIDSEARCH TO FIND BEST PARAMETERS
X_for_gridsearch = X_train[0:2000, :]
y_for_gridsearch = y_train[0:2000]

param_grid = [
  {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
]

# Uncomment below to perform GridSearchCV
# ------------------------------------------------------------------------------------------------
# est = SVC(kernel='rbf')

# grid = GridSearchCV(estimator=est, param_grid=param_grid[0], refit=True, verbose=3, n_jobs=-1)

# grid.fit(X_for_gridsearch, y_for_gridsearch)

# print("Best parameters according to GridSearchCV")
# print(grid.best_params_)
# ------------------------------------------------------------------------------------------------

print("\nBest hyperparameters found using GridSearchCV: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}")

# Train clf with discovered hyperparams
clf_rbf = SVC(kernel="rbf", C=10, gamma=0.01)

clf_rbf.fit(X_train, y_train)

prediction = clf_rbf.predict(X_test)

# Find correct predictions
correct = 0
for i,j in zip(prediction, y_test):
    if i == j:
        correct+=1

# Find accuracy
accuracy = correct/len(y_test) * 100
print(f"\nPrediction accuracy with SVC and best hyperparameters = {accuracy} %")

# ------------------------------------------------------------------------------------------------
# CREATE A ONE-VS-ALL CLASSIEFIER (SVM) AND COMPARE RESULTS WITH ONE-VS-ONE SKLEARN CLASSIFIER

# ONE-VS-ONE classiefier fit, predict and accuracy:
OvO = OneVsOneClassifier(clf_rbf)

OvO.fit(X_train, y_train)
y_pred = OvO.predict(X_test)

# Find correct predictions
count = 0
for i,j in zip(y_pred, y_test):
    if i == j:
        count+=1

# Find accuracy with built in OVA and print confusion matrix
accuracy = count/len(y_test) * 100
print(f"\nPrediction accuracy with built-in OVO: {accuracy} %")
print("\nConfusion matrix with built-in OVO:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------------------------------------------

clf_list = []

# Train 10 different classifiers, one for each target value
for i in range(10):

    # Binarize y_train based on what the target label currently is (0, 1, 2, 3,... , 9)
    y_train_binarized = np.where(y_train == i, 1, 0)
    
    # Fit classifier and append to a classifier list
    clf = SVC(C=10, gamma = 0.01, kernel= 'rbf')
    clf.fit(X_train, y_train_binarized)
    clf_list.append(clf)

# This creates 10 different predicted y lists
y_pred_list = []
correct = 0
for X_row, y in zip(X_test, y_test):

    y_pred = fs.one_vs_all(X_row, clf_list)

    y_pred_list.append(np.array(y_pred))

    if y_pred == y:
        correct += 1

# Transform the classifier list into a np array
y_pred_list = np.array(y_pred_list)

# Find accuracy with self implemented OVA and print confusion matrix
accuracy = round(correct / len(X_test) * 100, 1)
print(f"\nPrediction accuracy with self-implemented OVA: {accuracy} %")
print("\nConfusion matrix with self-implemented OVA:")
print(confusion_matrix(y_test, y_pred_list))
