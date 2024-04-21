import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

# ----------------------------------- READ ---------------------------------------
directory = os.path.dirname(os.path.abspath(__file__))

# microchips file contains 2 values followed by 1 (OK) or 0 (FAILED)
csv_file = open(os.path.join(directory, "data", "microchips.csv"), "r")

# csv.reader returns an iterable: csv_reader
csv_reader = csv.reader(csv_file, delimiter=',')

# create list of x values
X = []

# create list of y values
y = []

# fill the 2 lists with the values provided in the .csv files USING NUMPY
for i in csv_reader:
    a = np.array([float(i[0]), float(i[1])])
    b = np.array([float(i[2])])
    X.append(a)
    y.append(b)

X = np.array(X)
y = np.array(y)

# ----------------------------- FIND TRAINING ERRORS ---------------------------------

from sklearn.neighbors import KNeighborsClassifier

# create an error list containing the strings to be displayed on the plots
error_list = []

# find training errors for k values 1, 3, 5, 7
for i in range(8):

    if i % 2 == 1:

        clf1 = KNeighborsClassifier(n_neighbors=i) 
        clf1.fit(X, y.ravel())

        train_values_results = clf1.predict(X)

        counter = 0
        for j in range(len(train_values_results)):

            # compare the results with the original OK/FAIL values
            if train_values_results[j] != y[j]:
                counter += 1

        error_list.append(f"Errors with k({i}): {counter}")

# -------------------------- PREDICT 3 MICROCHIPS EXERCISE 1.2 ----------------------------

def print_errors(final_list, k):
    print()
    print(f"k = {k}")
    for i in range(len(final_list)):
        print(f"    chip{i+1}: {final_list[i][0]} ==> {final_list[i][1]}")
    print()

#VALUES FOR WHICH WE NEED A PREDICTION
test_values = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])

k = 1
clf2 = KNeighborsClassifier(n_neighbors=k) 

# assigning training values to the model
clf2.fit(X, y.ravel())
test_values_results = clf2.predict(test_values)


# list containing
final_list = []

for i, j in zip(test_values, test_values_results):
    if (j == 1.0):
        t = [i, "OK"]
        final_list.append(t)
    else:
        t = [i, "FAIL"]
        final_list.append(t)
print_errors(final_list, k)


k = 3
clf2 = KNeighborsClassifier(n_neighbors=k) 
# assigning training values to the model
clf2.fit(X, y.ravel())
test_values_results = clf2.predict(test_values)
# list containing
final_list = []

for i, j in zip(test_values, test_values_results):
    if (j == 1.0):
        t = [i, "OK"]
        final_list.append(t)
    else:
        t = [i, "FAIL"]
        final_list.append(t)
print_errors(final_list, k)

k = 5
clf2 = KNeighborsClassifier(n_neighbors=k) 
# assigning training values to the model
clf2.fit(X, y.ravel())
test_values_results = clf2.predict(test_values)
# list containing
final_list = []
for i, j in zip(test_values, test_values_results):
    if (j == 1.0):
        t = [i, "OK"]
        final_list.append(t)
    else:
        t = [i, "FAIL"]
        final_list.append(t)
print_errors(final_list, k)


k = 7
clf2 = KNeighborsClassifier(n_neighbors=k) 
# assigning training values to the model
clf2.fit(X, y.ravel())
test_values_results = clf2.predict(test_values)
# list containing
final_list = []

for i, j in zip(test_values, test_values_results):
    if (j == 1.0):
        t = [i, "OK"]
        final_list.append(t)
    else:
        t = [i, "FAIL"]
        final_list.append(t)
print_errors(final_list, k)
    

# ----------------------- GRID CREATION AND PLOT EXERCISE 1.1 AND 1.3 -----------------------

min_x, max_x = min(X[:, 0]), max(X[:, 0])
min_y, max_y = min(X[:, 1]), max(X[:, 1])
grid_size = 100
x_axis = np.linspace(min_x, max_x, grid_size)
y_axis = np.linspace(min_y, max_y, grid_size)

counter = 0
# create a grid with colored areas
def create_grid_with_k(i):
    clf3 = KNeighborsClassifier(n_neighbors=i)
    clf3.fit(X, y.ravel())

    # FAST FUNCTION FOR GRID
    def make_grid_opt(x_axis, y_axis):
        xx, yy = np.meshgrid(x_axis, y_axis)
        cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
        grid = clf3.predict(cells).reshape(grid_size, grid_size)
        return grid

    grid = make_grid_opt(x_axis, y_axis)
    return grid

cmap = ListedColormap(['#FF0000', '#00FF00'])

fig, (ax) = plt.subplots(3, 2, figsize=(11, 11))

ax[0, 0].scatter(X[:, 0], X[:, 1], c = y, edgecolors='k')
ax[0, 0].set_aspect('equal', 'box')
ax[0, 0].set_title("Original data")

ax[0, 1].imshow(create_grid_with_k(1), origin = 'lower', extent = (min_x, max_x, min_y, max_y))
ax[0, 1].scatter(X[:,0], X[:,1], c = y, cmap = cmap, edgecolors='k')
ax[0, 1].set_title(error_list[0])

ax[1, 0].imshow(create_grid_with_k(3), origin = 'lower', extent = (min_x, max_x, min_y, max_y))
ax[1, 0].scatter(X[:,0], X[:,1], c = y, cmap = cmap, edgecolors='k')
ax[1, 0].set_title(error_list[1])

ax[1, 1].imshow(create_grid_with_k(5), origin = 'lower', extent = (min_x, max_x, min_y, max_y))
ax[1, 1].scatter(X[:,0], X[:,1], c = y, cmap = cmap, edgecolors='k')
ax[1, 1].set_title(error_list[2])

ax[2, 0].imshow(create_grid_with_k(7), origin = 'lower', extent = (min_x, max_x, min_y, max_y))
ax[2, 0].scatter(X[:,0], X[:,1], c = y, cmap = cmap, edgecolors='k')
ax[2, 0].set_title(error_list[3])

ax[2, 1].axis('off')

plt.show()

