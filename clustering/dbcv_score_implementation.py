"""
Implementation of Density-Based Clustering Validation (DBCV)

Citation:
Moulavi, Davoud, et al. "Density-based clustering validation."
Proceedings of the 2014 SIAM International Conference on Data Mining.
Society for Industrial and Applied Mathematics, 2014.
"""


"""
Implementation based on: https://github.com/christopherjenness/DBCV/blob/master/DBCV/DBCV.py
I made it a bit faster, using: 
-precompute core distances
-precompute get label
-nested for loops are now vectorized
-smarter dense matrix construction

"""

import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
import time


def DBCV(X, labels, dist_function=euclidean):
    """
    Density-Based Clustering Validation (faster version).

    Args:
        X (np.ndarray): Data, shape (n_samples, n_features).
        labels (np.array): Clustering labels for data X.
        dist_function (function): Distance function, default: euclidean.

    Returns:
        cluster_validity (float): Score in range [-1, 1].
    """
    core_dists, label_members = _precompute_core_dists(X, labels, dist_function)
    graph = _mutual_reach_dist_graph_fast(X, core_dists, dist_function)
    mst = _mutual_reach_dist_MST(graph)
    cluster_validity = _clustering_validity_index(mst, labels)
    return cluster_validity


def _precompute_core_dists(X, labels, dist_function):
    """
    Precompute core distances for all points.
    """
    n_samples, n_features = X.shape
    core_dists = np.zeros(n_samples)
    label_members = {}

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        members = X[indices]
        label_members[label] = members

        dists = cdist(members, members)
        for idx, point_idx in enumerate(indices):
            distance_vector = dists[idx]
            distance_vector = distance_vector[distance_vector != 0]
            numerator = (1.0 / distance_vector ** n_features).sum()
            core_dist = (numerator / (len(members) - 1)) ** (-1.0 / n_features)
            core_dists[point_idx] = core_dist

    return core_dists, label_members


def _mutual_reach_dist_graph_fast(X, core_dists, dist_function):
    """
    Build mutual reachability distance graph (fast, symmetric).
    """
    n_samples = X.shape[0]
    graph = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = dist_function(X[i], X[j])
            mutual_reach = max(core_dists[i], core_dists[j], dist)
            graph[i, j] = mutual_reach
            graph[j, i] = mutual_reach

    return graph


def _mutual_reach_dist_MST(dist_graph):
    """
    Compute minimum spanning tree from distance graph.
    """
    mst = minimum_spanning_tree(dist_graph).toarray()
    return mst + mst.T


def _cluster_density_sparseness(MST, labels, cluster):
    """
    Compute minimum density (sparseness) inside a cluster.
    """
    indices = np.where(labels == cluster)[0]
    if len(indices) <= 1:
        return 0
    cluster_MST = MST[np.ix_(indices, indices)]
    return np.max(cluster_MST)


def _cluster_density_separation(MST, labels, cluster_i, cluster_j):
    """
    Compute density separation between two clusters.
    """
    indices_i = np.where(labels == cluster_i)[0]
    indices_j = np.where(labels == cluster_j)[0]

    if len(indices_i) == 0 or len(indices_j) == 0:
        return np.inf

    shortest_paths = dijkstra(MST, indices=indices_i)
    relevant_paths = shortest_paths[:, indices_j]
    return np.min(relevant_paths)


def _cluster_validity_index(MST, labels, cluster):
    """
    Compute validity index for a single cluster.
    """
    min_density_separation = np.inf

    for other_cluster in np.unique(labels):
        if other_cluster != cluster:
            separation = _cluster_density_separation(MST, labels, cluster, other_cluster)
            if separation < min_density_separation:
                min_density_separation = separation

    cluster_sparseness = _cluster_density_sparseness(MST, labels, cluster)

    numerator = min_density_separation - cluster_sparseness
    denominator = max(min_density_separation, cluster_sparseness)

    if denominator == 0:
        return 0

    return numerator / denominator


def _clustering_validity_index(MST, labels):
    """
    Compute overall clustering validity.
    """
    n_samples = len(labels)
    validity = 0.0

    for cluster in np.unique(labels):
        fraction = np.sum(labels == cluster) / n_samples
        cluster_validity = _cluster_validity_index(MST, labels, cluster)
        validity += fraction * cluster_validity

    return validity
