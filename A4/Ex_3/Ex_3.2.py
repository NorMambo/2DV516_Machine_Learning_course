import matplotlib.pyplot as plt
import numpy as np
import os
import Functions as fs
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering

# NOTE: The program is very slow!!

# Paths to data files
dirname = os.path.dirname(os.path.abspath(__file__))
path1 = os.path.join(dirname, 'data', 'phoneme.csv')
path2 = os.path.join(dirname, 'data', 'pollen.csv')
path3 = os.path.join(dirname, 'data', 'run_or_walk.csv')

# Generate datasets as np arrays
phoneme = np.genfromtxt(fname=path1, delimiter=',', skip_header=1)
pollen = np.genfromtxt(fname=path2, delimiter=',', skip_header=1)
run_or_walk = np.genfromtxt(fname=path3, delimiter=',', skip_header=1)

# X for every dataset.
X_phoneme = phoneme[:, :5]
X_pollen = pollen[:, :5]
X_run_or_walk = run_or_walk[:, :6]

# components = dimension of data
tsne = TSNE(n_components=2)

# Redimensioned data
redim_phoneme = tsne.fit_transform(X_phoneme)
redim_pollen = tsne.fit_transform(X_pollen)
redim_run_or_walk = tsne.fit_transform(X_run_or_walk)

# Lists for plot
clustered_data_list = []
reduced_X_list = [redim_pollen, redim_pollen, redim_pollen,
                  redim_run_or_walk, redim_run_or_walk, redim_run_or_walk,
                  redim_phoneme, redim_phoneme, redim_phoneme]

# POLLEN: labels are 2
bkmeans_pollen = fs.bisecting_kmeans(X=redim_pollen,k=2, iter=10)
clustered_data_list.append(bkmeans_pollen)
kmeans_pollen = KMeans(n_clusters=2)
kmeans_pollen.fit(redim_pollen)
clustered_data_list.append(kmeans_pollen.labels_)
hac_pollen = AgglomerativeClustering(n_clusters=2)
hac_pollen.fit(redim_pollen)
clustered_data_list.append(hac_pollen.labels_)

# # RUN OR WALK: labels are 2
bkmeans_run_walk = fs.bisecting_kmeans(X=redim_run_or_walk,k=2, iter=10)
clustered_data_list.append(bkmeans_run_walk)
kmeans_run_walk = KMeans(n_clusters=2)
kmeans_run_walk.fit(redim_run_or_walk)
clustered_data_list.append(kmeans_run_walk.labels_)
hac_run_walk = AgglomerativeClustering(n_clusters=2)
hac_run_walk.fit(redim_run_or_walk)
clustered_data_list.append(hac_run_walk.labels_)

# PHONEME: labels are 2
bkmeans_Phoneme = fs.bisecting_kmeans(X=redim_phoneme,k=2, iter=10)
clustered_data_list.append(bkmeans_Phoneme)
kmeans_Phoneme = KMeans(n_clusters=2)
kmeans_Phoneme.fit(redim_phoneme)
clustered_data_list.append(kmeans_Phoneme.labels_)
hac_phoneme = AgglomerativeClustering(n_clusters=2)
hac_phoneme.fit(redim_phoneme)
clustered_data_list.append(hac_phoneme.labels_)

figure_lables = ["BKmeans Pollen",
                "Kmeans Pollen",
                "HAC Pollen",
                "BKmeans Run Or Walk",
                "Kmeans Run Or Walk",
                "HAC Run Or Walk",
                "BKmeans Phoneme",
                "Kmeans Phoneme",
                "HAC Phoneme"]

cmap_l = ['spring', 'rainbow', 'winter']

plt.figure(figsize=(12, 10))
for i in range(9):  
    plt.subplot(3, 3, i + 1)
    plt.scatter(reduced_X_list[i][:, 0], reduced_X_list[i][:, 1], c=clustered_data_list[i], s=2, cmap=cmap_l[i%3])
    plt.title(figure_lables[i])
plt.show()