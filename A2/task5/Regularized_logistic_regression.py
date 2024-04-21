import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import Functions as fs

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory, "microchips.csv")
data = genfromtxt(path, delimiter=',')

X_train = np.array(data[:, :2])
y_train = np.array(data[:, 2])
n = X_train.shape[0]

X_train_positive = []
X_train_negative = []
for row in data:
    if row[2] == 1:
        X_train_positive.append(row)
    else:
        X_train_negative.append(row)
X_train_negative = np.array(X_train_negative)
X_train_positive = np.array(X_train_positive)

extended_X_trains = []

deg_1_train = fs.map_feature(X_train[:, 0], X_train[:, 1], 1)
deg_2_train = fs.map_feature(X_train[:, 0], X_train[:, 1], 2)
deg_3_train = fs.map_feature(X_train[:, 0], X_train[:, 1], 3)
deg_4_train = fs.map_feature(X_train[:, 0], X_train[:, 1], 4)
deg_5_train = fs.map_feature(X_train[:, 0], X_train[:, 1], 5)
deg_6_train = fs.map_feature(X_train[:, 0], X_train[:, 1], 6)
deg_7_train = fs.map_feature(X_train[:, 0], X_train[:, 1], 7)
deg_8_train = fs.map_feature(X_train[:, 0], X_train[:, 1], 8)
deg_9_train = fs.map_feature(X_train[:, 0], X_train[:, 1], 9)

extended_X_trains.append(deg_1_train)
extended_X_trains.append(deg_2_train)
extended_X_trains.append(deg_3_train)
extended_X_trains.append(deg_4_train)
extended_X_trains.append(deg_5_train)
extended_X_trains.append(deg_6_train)
extended_X_trains.append(deg_7_train)
extended_X_trains.append(deg_8_train)
extended_X_trains.append(deg_9_train)

# 1) Use Logistic regression and mapFeatures from the previous exercise to construct
# nine different classifiers, one for each of the degrees d 2 [1; 9], and produce
# a figure containing a 3 X 3 pattern of subplots showing the corresponding decision boundaries.
# Make sure that you pass the argument C=10000.

# 2) Redo 1) but now use the regularization parameter C = 1.
# What is different than from the step in 1)?

# NOTE: change the C parameter here:

clf_1_list = []
C1 = 1
for Xe in extended_X_trains:
    clf = LogisticRegression(solver='lbfgs', C = C1, tol = 1e-6, max_iter = 5000).fit(Xe, y_train)
    clf_1_list.append(clf)


clf_10000_list = []
C2 = 10000
for Xe in extended_X_trains:
    clf = LogisticRegression(solver='lbfgs', C = C2, tol = 1e-6, max_iter = 5000).fit(Xe, y_train)
    clf_10000_list.append(clf)


error_1_list = []
for clf, Xe in zip(clf_1_list, extended_X_trains):
    pred_y = clf.predict(Xe)
    pred_y = np.reshape(pred_y, (-1, 1))
    errors = np.sum(pred_y!=np.reshape(y_train, (-1, 1)))
    error_1_list.append(errors)


error_10000_list = []
for clf, Xe in zip(clf_10000_list, extended_X_trains):
    pred_y = clf.predict(Xe)
    pred_y = np.reshape(pred_y, (-1, 1))
    errors = np.sum(pred_y!=np.reshape(y_train, (-1, 1)))
    error_10000_list.append(errors)

# -------------------------------------------- PLOT C = 10000 ----------------------------------------------------

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig = pylab.gcf()
fig.canvas.manager.set_window_title('C = 10000')

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_10000_list[0], 1)
axs[0][0].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[0][0].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[0][0].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[0][0].set_title(f"Training Errors: {error_10000_list[0]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_10000_list[1], 2)
axs[0][1].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[0][1].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[0][1].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[0][1].set_title(f"Training Errors: {error_10000_list[1]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_10000_list[2], 3)
axs[0][2].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[0][2].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[0][2].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[0][2].set_title(f"Training Errors: {error_10000_list[2]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_10000_list[3], 4)
axs[1][0].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[1][0].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[1][0].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[1][0].set_title(f"Training Errors: {error_10000_list[3]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_10000_list[4], 5)
axs[1][1].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[1][1].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[1][1].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[1][1].set_title(f"Training Errors: {error_10000_list[4]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_10000_list[5], 6)
axs[1][2].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[1][2].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[1][2].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[1][2].set_title(f"Training Errors: {error_10000_list[5]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_10000_list[6], 7)
axs[2][0].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[2][0].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[2][0].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[2][0].set_title(f"Training Errors: {error_10000_list[6]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_10000_list[7], 8)
axs[2][1].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[2][1].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[2][1].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[2][1].set_title(f"Training Errors: {error_10000_list[7]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_10000_list[8], 9)
axs[2][2].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[2][2].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[2][2].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[2][2].set_title(f"Training Errors: {error_10000_list[8]}")

plt.show()

# ------------------------------------------ PLOT C = 1 ----------------------------------------------------

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig = pylab.gcf()
fig.canvas.manager.set_window_title('C = 1')

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_1_list[0], 1)
axs[0][0].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[0][0].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[0][0].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[0][0].set_title(f"Training Errors: {error_1_list[0]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_1_list[1], 2)
axs[0][1].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[0][1].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[0][1].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[0][1].set_title(f"Training Errors: {error_1_list[1]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_1_list[2], 3)
axs[0][2].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[0][2].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[0][2].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[0][2].set_title(f"Training Errors: {error_1_list[2]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_1_list[3], 4)
axs[1][0].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[1][0].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[1][0].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[1][0].set_title(f"Training Errors: {error_1_list[3]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_1_list[4], 5)
axs[1][1].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[1][1].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[1][1].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[1][1].set_title(f"Training Errors: {error_1_list[4]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_1_list[5], 6)
axs[1][2].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[1][2].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[1][2].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[1][2].set_title(f"Training Errors: {error_1_list[5]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_1_list[6], 7)
axs[2][0].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[2][0].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[2][0].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[2][0].set_title(f"Training Errors: {error_1_list[6]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_1_list[7], 8)
axs[2][1].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[2][1].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[2][1].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[2][1].set_title(f"Training Errors: {error_1_list[7]}")

xx, yy, clz_mesh, cmap_light, cmap_bold = fs.create_decision_boundary(X_train, clf_1_list[8], 9)
axs[2][2].pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
axs[2][2].scatter(X_train_positive[:, 0], X_train_positive[:, 1], c="Green" ,marker="." ,cmap=cmap_bold)
axs[2][2].scatter(X_train_negative[:, 0], X_train_negative[:, 1], c="Red" ,marker="." ,cmap=cmap_bold)
axs[2][2].set_title(f"Training Errors: {error_1_list[8]}")

plt.show()



# 3) Finally, you should use cross-validation (in sklearn) to see which of the regularized and unregularized models performs best.
# The results could for instance be visualized in a graph where you plot the degree d vs. #errors, and differentiate regularized and unregularized by color.

score_list_c1 = []
score_list_c10000 = []
error_list_c1 = []
error_list_c10000 = []

k = 5

score_1 = cross_val_score(clf_1_list[0], deg_1_train, y_train, cv = k, scoring="accuracy")
score_1_mean = np.mean(score_1)
errors_1 = round(n - (score_1_mean * n))
error_list_c1.append(errors_1)
score_list_c1.append(score_1_mean)

score_2 = cross_val_score(clf_1_list[1], deg_2_train, y_train, cv = k, scoring="accuracy")
score_2_mean = np.mean(score_2)
errors_2 = round(n - (score_2_mean * n))
error_list_c1.append(errors_2)
score_list_c1.append(score_2_mean)

score_3 = cross_val_score(clf_1_list[2], deg_3_train, y_train, cv = k, scoring="accuracy")
score_3_mean = np.mean(score_3)
errors_3 = round(n - (score_3_mean * n))
error_list_c1.append(errors_3)
score_list_c1.append(score_3_mean)

score_4 = cross_val_score(clf_1_list[3], deg_4_train, y_train, cv = k, scoring="accuracy")
score_4_mean = np.mean(score_4)
errors_4 = round(n - (score_4_mean * n))
error_list_c1.append(errors_4)
score_list_c1.append(score_4_mean)

score_5 = cross_val_score(clf_1_list[4], deg_5_train, y_train, cv = k, scoring="accuracy")
score_5_mean = np.mean(score_5)
errors_5 = round(n - (score_5_mean * n))
error_list_c1.append(errors_5)
score_list_c1.append(score_5_mean)

score_6 = cross_val_score(clf_1_list[5], deg_6_train, y_train, cv = k, scoring="accuracy")
score_6_mean = np.mean(score_6)
errors_6 = round(n - (score_6_mean * n))
error_list_c1.append(errors_6)
score_list_c1.append(score_6_mean)

score_7 = cross_val_score(clf_1_list[6], deg_7_train, y_train, cv = k, scoring="accuracy")
score_7_mean = np.mean(score_7)
errors_7 = round(n - (score_7_mean * n))
error_list_c1.append(errors_7)
score_list_c1.append(score_7_mean)

score_8 = cross_val_score(clf_1_list[7], deg_8_train, y_train, cv = k, scoring="accuracy")
score_8_mean = np.mean(score_8)
errors_8 = round(n - (score_8_mean * n))
error_list_c1.append(errors_8)
score_list_c1.append(score_8_mean)

score_9 = cross_val_score(clf_1_list[8], deg_9_train, y_train, cv = k, scoring="accuracy")
score_9_mean = np.mean(score_9)
errors_9 = round(n - (score_9_mean * n))
error_list_c1.append(errors_9)
score_list_c1.append(score_9_mean)

# --------------------------------------------------------------------------------------

score_1 = cross_val_score(clf_10000_list[0], deg_1_train, y_train, cv = k, scoring="accuracy")
score_1_mean = np.mean(score_1)
errors_1 = round(n - (score_1_mean * n))
error_list_c10000.append(errors_1)
score_list_c10000.append(score_1_mean)

score_2 = cross_val_score(clf_10000_list[1], deg_2_train, y_train, cv = k, scoring="accuracy")
score_2_mean = np.mean(score_2)
errors_2 = round(n - (score_2_mean * n))
error_list_c10000.append(errors_2)
score_list_c10000.append(score_2_mean)

score_3 = cross_val_score(clf_10000_list[2], deg_3_train, y_train, cv = k, scoring="accuracy")
score_3_mean = np.mean(score_3)
errors_3 = round(n - (score_3_mean * n))
error_list_c10000.append(errors_3)
score_list_c10000.append(score_3_mean)

score_4 = cross_val_score(clf_10000_list[3], deg_4_train, y_train, cv = k, scoring="accuracy")
score_4_mean = np.mean(score_4)
errors_4 = round(n - (score_4_mean * n))
error_list_c10000.append(errors_4)
score_list_c10000.append(score_4_mean)

score_5 = cross_val_score(clf_10000_list[4], deg_5_train, y_train, cv = k, scoring="accuracy")
score_5_mean = np.mean(score_5)
errors_5 = round(n - (score_5_mean * n))
error_list_c10000.append(errors_5)
score_list_c10000.append(score_5_mean)

score_6 = cross_val_score(clf_10000_list[5], deg_6_train, y_train, cv = k, scoring="accuracy")
score_6_mean = np.mean(score_6)
errors_6 = round(n - (score_6_mean * n))
error_list_c10000.append(errors_6)
score_list_c10000.append(score_6_mean)

score_7 = cross_val_score(clf_10000_list[6], deg_7_train, y_train, cv = k, scoring="accuracy")
score_7_mean = np.mean(score_7)
errors_7 = round(n - (score_7_mean * n))
error_list_c10000.append(errors_7)
score_list_c10000.append(score_7_mean)

score_8 = cross_val_score(clf_10000_list[7], deg_8_train, y_train, cv = k, scoring="accuracy")
score_8_mean = np.mean(score_8)
errors_8 = round(n - (score_8_mean * n))
error_list_c10000.append(errors_8)
score_list_c10000.append(score_8_mean)

score_9 = cross_val_score(clf_10000_list[8], deg_9_train, y_train, cv = k, scoring="accuracy")
score_9_mean = np.mean(score_9)
errors_9 = round(n - (score_9_mean * n))
error_list_c10000.append(errors_9)
score_list_c10000.append(score_9_mean)

degree_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

plt.plot(degree_list, error_list_c1, color="Green", label="C1")
plt.plot(degree_list, error_list_c10000, color="Red", label="C10000")
plt.legend()
plt.xlabel("DEGREE")
plt.ylabel("ERRORS")

plt.show()


