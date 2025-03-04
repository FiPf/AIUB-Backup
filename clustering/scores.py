from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import cdist

from clustering_utils import ClusteringResult
from collections import namedtuple
from typing import Callable
from scipy.spatial import ConvexHull
import numpy as np

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html
#https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index 
def DB_score(ClusteringResult: namedtuple): 
    data = ClusteringResult.data 
    labels = ClusteringResult.labels

    score = davies_bouldin_score(data, labels)

    return score

#https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html 
def CH_score(ClusteringResult: namedtuple): 
    data = ClusteringResult.data 
    labels = ClusteringResult.labels

    score = calinski_harabasz_score(data, labels)

    return score

#https://en.wikipedia.org/wiki/Dunn_index
#https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/
def dunn_index_score(ClusteringResult: namedtuple):
    data = ClusteringResult.data
    labels = ClusteringResult.labels
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return -1  # Dunn index is not meaningful for a single cluster

    clusters = [data[labels == label] for label in unique_labels]

    # Compute intra-cluster distances (maximum within-cluster distance)
    intra_cluster_distances = [np.max(cdist(cluster, cluster)) for cluster in clusters if len(cluster) > 1]
    
    # Compute inter-cluster distances (minimum distance between any two clusters)
    inter_cluster_distances = [
        np.min(cdist(clusters[i], clusters[j]))
        for i in range(len(clusters)) for j in range(i + 1, len(clusters))
    ]

    if not intra_cluster_distances or not inter_cluster_distances:
        return -1  # Avoid division by zero

    return min(inter_cluster_distances) / max(intra_cluster_distances)

#silhouette score
def sil_score(ClusteringResult: namedtuple):
    data = ClusteringResult.data
    labels = ClusteringResult.labels

    # Calculate silhouette score (a higher value means better clustering)
    if len(np.unique(labels)) > 1:  # Ensure there is more than one cluster
        score = silhouette_score(data, labels)
    else:
        score = -1  # Return a negative score if clustering fails
        
    num_clusters = len(np.unique(labels))
    if num_clusters < 5: # at least 5 clusters
        score -= (5 - num_clusters) * 0.1  # Penalty for too few clusters
        
    return score

#2d standard deviation
#calculate the covariance matrix and its eigenvalues. useful for clusters that are not axis aligned or elongated
def cluster_std_eigen(data, labels):
    unique_labels = np.unique(labels)
    stds = {}

    for label in unique_labels:
        cluster_points = data[labels == label]  # Select points in this cluster
        if len(cluster_points) < 2:
            stds[label] = (0, 0)  # Avoid issues with single-point clusters
            continue

        # Compute covariance matrix
        cov_matrix = np.cov(cluster_points.T)  # Transpose to match np.cov input format
        
        # Compute eigenvalues (variances along principal axes)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)

        # Standard deviations are the square roots of the eigenvalues
        stds[label] = tuple(np.sqrt(eigenvalues))

    return stds

def cluster_density_squares(ClusteringResult: namedtuple):
    #idea: iterate over all clusters. 
    #take the minima and maxima of the cluster coordinates in both x and y directions
    #take this values to form a square, calculate the area of that square
    #take the number of clusters within the square and find the cluster denisty based on the square area
    data = ClusteringResult.data
    labels = ClusteringResult.labels
    unique_labels = np.unique(labels)
    densities = {}
    boundaries = {}

    for label in unique_labels:
        cluster_points = data[labels == label]
        min_x, min_y = np.min(cluster_points, axis=0)
        max_x, max_y = np.max(cluster_points, axis=0)

        square_area = (max_x - min_x) * (max_y - min_y)
        num_points = len(cluster_points)

        density = num_points / square_area if square_area > 0 else np.inf
        densities[label] = density
        boundaries[label] = ((min_x, min_y), (max_x, max_y))

    return densities, boundaries

def cluster_density_convex_hull(ClusteringResult: namedtuple):
    #idea: iterate over all clusters. 
    #take the minima and maxima of the cluster coordinates in both x and y directions
    #take this values to form the convex hull, calculate the area of that convex hull (use hull.volume, more efficient than Monte Carlo)
    #take the number of clusters within the convec hull and find the cluster denisty based on the convex hull area
    data = ClusteringResult.data
    labels = ClusteringResult.labels
    unique_labels = np.unique(labels)
    densities = {}
    boundaries = {}

    for label in unique_labels:
        cluster_points = data[labels == label]

        if len(cluster_points) < 3:
            densities[label] = np.inf  # Convex hull is not defined for <3 points
            boundaries[label] = None
            continue

        hull = ConvexHull(cluster_points)
        hull_area = hull.volume  # In 2D, hull.volume gives the area
        num_points = len(cluster_points)

        density = num_points / hull_area if hull_area > 0 else np.inf
        densities[label] = density
        boundaries[label] = cluster_points[hull.vertices]  # Boundary points forming the convex hull

    return densities, boundaries