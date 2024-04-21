import matplotlib.pyplot as plt
import numpy as np
import os
import Functions as fs
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths to data files
dirname = os.path.dirname(os.path.abspath(__file__))
path1 = os.path.join(dirname, 'data', 'phoneme.csv')
path2 = os.path.join(dirname, 'data', 'pollen.csv')
path3 = os.path.join(dirname, 'data', 'run_or_walk.csv')

# Generate datasets as np arrays
phoneme = np.genfromtxt(fname=path1, delimiter=',', skip_header=1)
pollen = np.genfromtxt(fname=path2, delimiter=',', skip_header=1)
run_or_walk = np.genfromtxt(fname=path3, delimiter=',', skip_header=1)

# Get X for every dataset
X_phoneme = phoneme[:, :5]
X_pollen = pollen[:, :5]
X_run_or_walk = run_or_walk[:, :6]
# Get y for every dataset
y_phoneme = phoneme[:, 5:]
y_pollen = pollen[:, 5:]
y_run_or_walk = run_or_walk[:, 6:]

# List for plot
list_of_y_sets = [y_pollen, y_pollen, y_pollen,
                  y_run_or_walk, y_run_or_walk, y_run_or_walk,
                  y_phoneme, y_phoneme, y_phoneme,]

# Normalize features for PCA and Sammon
scaler = StandardScaler()
scaler.fit(X_phoneme)
norm_X_phoneme = scaler.transform(X_phoneme)
scaler.fit(X_pollen)
norm_X_pollen = scaler.transform(X_pollen)
scaler.fit(run_or_walk)
norm_X_run_walk = scaler.transform(run_or_walk)

list_of_X_sets = []

# POLLEN
X_pricted_1 = fs.sammon(norm_X_pollen, 100, 0.2, 0.001)
list_of_X_sets.append(X_pricted_1)
pca = PCA(n_components=2)
pca.fit(norm_X_pollen)
X_pricted_2 = pca.transform(norm_X_pollen)
list_of_X_sets.append(X_pricted_2)
tsne = TSNE(n_components=2)
X_pricted_3 = tsne.fit_transform(X_pollen)
list_of_X_sets.append(X_pricted_3)

# RUN OR WALK
X_pricted_4 = fs.sammon(norm_X_run_walk, 100, 0.2, 0.001)
list_of_X_sets.append(X_pricted_4)
pca = PCA(n_components=2)
pca.fit(norm_X_run_walk)
X_pricted_5 = pca.transform(norm_X_run_walk)
list_of_X_sets.append(X_pricted_5)
tsne = TSNE(n_components=2)
X_predicted_6 = tsne.fit_transform(X_run_or_walk)
list_of_X_sets.append(X_predicted_6)

# PHONEME
X_pricted_7 = fs.sammon(norm_X_phoneme, 100, 0.2, 0.001) # original l_rate 0.3
list_of_X_sets.append(X_pricted_7)
pca = PCA(n_components=2)
pca.fit(norm_X_phoneme)
X_predicted_8 = pca.transform(norm_X_phoneme)
list_of_X_sets.append(X_predicted_8)
tsne = TSNE(n_components=2)
X_pricted_9 = tsne.fit_transform(X_phoneme)
list_of_X_sets.append(X_pricted_9)


figure_lables = ["Sammon Pollen",
                "PCA Pollen",
                "TSNE Pollen",
                "Sammon Run or Walk",
                "PCA Run or Walk",
                "TSNE Run or Walk",
                "Sammon Phoneme",
                "PCA Phoneme",
                "TSNE Phoneme"]

cmap_l = ['spring', 'rainbow', 'winter']
# cmap_l = ['cool', 'winter']
plt.figure(figsize=(12, 10))
for i in range(9):  
    plt.subplot(3, 3, i + 1)
    plt.scatter(list_of_X_sets[i][:, 0], list_of_X_sets[i][:, 1], c=list_of_y_sets[i], s=2, cmap=cmap_l[i%3])
    plt.title(figure_lables[i])
plt.show()