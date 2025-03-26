from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import cdist

from clustering_utils import ClusteringResult
from collections import namedtuple
from typing import Callable
from scipy.spatial import ConvexHull, qhull
import numpy as np

def DB_score(ClusteringResult: namedtuple): 
    """measures the average similarity ratio of each cluster with its most similar cluster, taking into account:
    a) intra-cluster similarity (compactness)
    b) inter-cluster difference (separation)

    Lower value = better. Values below 1 indicate a good separation and compactness. 

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html
    https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index 
    Args:
        ClusteringResult (namedtuple): results from the clustering containing labels, cluster_centers and data.

    Returns:
        score (float): Davies Boulidin Score of the clustered data
    """
    data = ClusteringResult.data 
    labels = ClusteringResult.labels

    score = davies_bouldin_score(data, labels)

    return score

def CH_score(ClusteringResult: namedtuple):
    """gives the ratio between
    -between cluster separation (weighted sum of squared euclidian distance between cluster points and centroids)
    -within cluster dispersion (sum of squared Euclidian distances between data points and cluster centroids)
    normalized by the number of degrees of freedom

    Higher score = better.

    https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html 

    Args:
        ClusteringResult (namedtuple): results from the clustering containing labels, cluster_centers and data.

    Returns:
        score (float): Calinski Harabasz Score of the data. 
    """
    data = ClusteringResult.data 
    labels = ClusteringResult.labels

    score = calinski_harabasz_score(data, labels)

    return score

def dunn_index_score(ClusteringResult: namedtuple):
    """evaluates the compactness and separation of clusters. It is defined as the ratio between the **minimum inter-cluster distance** 
    (the smallest distance between points in different clusters) and the **maximum intra-cluster distance** 
    (the largest distance within any single cluster).  

    \[
    D = \frac{\min_{i \neq j} d(C_i, C_j)}{\max_k d(C_k)}
    \]

    Where:  
    - \( d(C_i, C_j) \) is the distance between clusters \( C_i \) and \( C_j \).  
    - \( d(C_k) \) is the maximum intra-cluster distance for cluster \( C_k \).  

    Higher dunn index = better. 

    https://en.wikipedia.org/wiki/Dunn_index
    https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/

    Args:
        ClusteringResult (namedtuple): results from the clustering containing labels, cluster_centers and data.

    Returns:
        (float): dunn index score of the data
        
    """
    data = ClusteringResult.data
    labels = ClusteringResult.labels
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return -1  

    clusters = [data[labels == label] for label in unique_labels]

    intra_cluster_distances = [np.max(cdist(cluster, cluster)) for cluster in clusters if len(cluster) > 1]
    
    inter_cluster_distances = [
        np.min(cdist(clusters[i], clusters[j]))
        for i in range(len(clusters)) for j in range(i + 1, len(clusters))
    ]

    if not intra_cluster_distances or not inter_cluster_distances:
        return -1  

    return min(inter_cluster_distances) / max(intra_cluster_distances)

def sil_score(ClusteringResult: namedtuple):
    """measures how well-separated and compact clusters are, ranging from -1 to 1.  

    It is defined as:  
    \[
    S = \frac{b - a}{\max(a, b)}
    \]
    where:  
    - \( a \) is the **average intra-cluster distance** (how close a point is to other points in its own cluster).  
    - \( b \) is the **average nearest-cluster distance** (how close a point is to the nearest other cluster).  

    Interpretation:
    - \( S \approx 1 \) → Well-clustered, points are far from other clusters.  
    - \( S \approx 0 \) → Overlapping clusters.  
    - \( S \approx -1 \) → Poor clustering, points may be in the wrong cluster.  

    Higher score = better

    Args:
        ClusteringResult (namedtuple): results from the clustering containing labels, cluster_centers and data.

    Returns:
        score (float): silhouette score of the data
    """
    data = ClusteringResult.data
    labels = ClusteringResult.labels

    if len(np.unique(labels)) > 1:  
        score = silhouette_score(data, labels)
    else:
        score = -1  
        
    num_clusters = len(np.unique(labels))
    if num_clusters < 5:
        score -= (5 - num_clusters) * 0.1  
        
    return score

def cluster_std_eigen(ClusteringResult: namedtuple):
    """2d standard deviation
    calculate the covariance matrix and its eigenvalues. useful for clusters that are not axis aligned or elongated.

    Args:
        ClusteringResult (namedtuple): results from the clustering containing labels, cluster_centers and data.

    Returns:
        stds (np.array): standard deviation for each dimension of the data
    """
    data = ClusteringResult.data
    labels = ClusteringResult.labels
    unique_labels = np.unique(labels)
    stds = {}

    for label in unique_labels:
        cluster_points = data[labels == label]  # Select points in this cluster
        if len(cluster_points) < 2:
            stds[label] = (0, 0)  # Avoid issues with single-point clusters
            continue

        # Covariance matrix
        cov_matrix = np.cov(cluster_points.T)  # Transpose to match np.cov input format
        
        # Compute eigenvalues (= variances along principal axes)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)

        # Standard deviations are the square roots of the eigenvalues
        stds[label] = tuple(np.sqrt(eigenvalues))

    return stds

def cluster_density_squares(ClusteringResult: namedtuple):
    """Goal: density of the cluster assuming the cluster shape is a square
    iterate over all clusters. 
    take the minima and maxima of the cluster coordinates in both x and y directions
    take this values to form a square, calculate the area of that square
    take the number of clusters within the square and find the cluster denisty based on the square area

    Args:
        ClusteringResult (namedtuple): results from the clustering containing labels, cluster_centers and data.

    Returns:
        densities (dict): dictionary containing the density for each cluster
        boundaries (dict): dictionary containing the boundaries for each cluster
    """    
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

def cluster_density_convex_hull(ClusteringResult):
    """Goal: density of the cluster and using the convex hull of the cluster as the shape
    iterate over all clusters. 
    use convex hull from a library to get the cluster shapes. Estimate the area of that convex hull.
    take the number of clusters within the convex hull and calculate the density.

    Args:
        ClusteringResult (namedtuple): results from the clustering containing labels, cluster_centers and data.

    Returns:
        densities (dict): dictionary containing the density for each cluster
        boundaries (dict): dictionary containing the boundaries for each cluster
    """    
    data = ClusteringResult.data
    labels = ClusteringResult.labels
    unique_labels = np.unique(labels)
    densities = {}
    boundaries = {}

    for label in unique_labels:
        cluster_points = data[labels == label]

        #print(f"Processing cluster {label} with {len(cluster_points)} points")

        if len(cluster_points) < 3:
            #print(f"Skipping cluster {label}: only {len(cluster_points)} points (need at least 3)")
            densities[label] = np.inf
            boundaries[label] = None
            continue
        
        if np.all(cluster_points == cluster_points[0]):  
            #print(f"Skipping cluster {label}: all points are identical")
            densities[label] = np.inf
            boundaries[label] = None
            continue

        try:
            hull = ConvexHull(cluster_points)
            hull_area = hull.volume  # In 2D, hull.volume gives the area
            num_points = len(cluster_points)

            density = num_points / hull_area if hull_area > 0 else np.inf
            densities[label] = density
            boundaries[label] = cluster_points[hull.vertices]  # Boundary points forming the convex hull

            #print(f"Cluster {label}: hull area = {hull_area}, density = {density}")

        except qhull.QhullError as e:
            #print(f"ConvexHull failed for cluster {label}: {e}")
            densities[label] = np.inf
            boundaries[label] = None  # Handle error 

    return densities, boundaries
