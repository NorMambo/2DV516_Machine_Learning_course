import numpy as np
from sklearn.cluster import KMeans

def bisecting_kmeans(X, k, iter):

    # where to store the result
    result_cluster = np.zeros(len(X), dtype=int)

    # initialize min_SSE
    min_SSE = 0

    # assign X to the variable that will be updated in the loops
    cluster_to_fit = X

    for i in range(1, k):
        for j in range(iter):

            # fit and find the SSE for the bisection
            k_means = KMeans(n_clusters=2, n_init=10)
            k_means.fit(cluster_to_fit)
            SSE = k_means.inertia_

            # find cluster indexes for the round
            if j == 0:
                min_SSE = SSE
                cluster_indexes = k_means.labels_
            else:
                if SSE < min_SSE:
                    min_SSE = SSE
                    cluster_indexes = k_means.labels_

        # create a copy of the result cluster where the result cluster has label zero
        # remaining_free_spots will always just be an array of zeros, but it will decrease in size
        remaining_free_spots = np.copy(result_cluster[result_cluster == 0])

        # create the subclusters
        subcluster1 = cluster_to_fit[cluster_indexes == 1]
        subcluster2 = cluster_to_fit[cluster_indexes == 0]

        # find SSE's
        sse1 = np.sum(np.square(subcluster1 - np.mean(subcluster1)))
        sse2 = np.sum(np.square(subcluster2 - np.mean(subcluster2)))

        # if SSE1 (of subcluster 1) is greater, fit Kmeans with subcluster1 in the next round
        if (sse1 > sse2):
            cluster_to_fit = subcluster1

            # let the label assigned to subcluster2 (lower SSE) take the spots in remaining_free_spots
            # note that the zeros will not be affected
            remaining_free_spots[cluster_indexes == 0] = i

        # if SSE2 (of subcluster 2) is greater, fit Kmeans with subcluster2 in the next round
        else:
            cluster_to_fit = subcluster2

            # let the label assigned to subcluster 1 (lower SSE) take the spots in remaining_free_spots
            remaining_free_spots[cluster_indexes == 1] = i

        # where result_cluster == 0, replace with the full layout of remaining_free_spots
        # remaining_free_spots and result_cluster[result_cluse == 0] have the same shape!
        result_cluster[result_cluster == 0] = remaining_free_spots


    return result_cluster
