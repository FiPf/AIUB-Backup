from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import cdist

from clustering_utils import ClusteringResult
from collections import namedtuple
from typing import Callable
from scipy.spatial import ConvexHull, qhull
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.cm import get_cmap

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

    dunn = min(inter_cluster_distances) / max(intra_cluster_distances)
    return dunn

def normalize_score(score, min_score, max_score):
    return (score - min_score) / (max_score - min_score) if max_score != min_score else 0.0

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

def plot_scores_for_different_binnings(array_of_metrics, array_of_yearranges, array_of_binwidths, store_dir):
    score_names = ["Davies-Bouldin", "Calinski-Harabasz", "Dunn Index", "Silhouette Score"]
    os.makedirs(store_dir, exist_ok=True)

    # Prepare data structure: score -> list of (year label, score value, bin width)
    metrics_per_score = {score: [] for score in score_names}
    
    for metrics, year_range, binwidth in zip(array_of_metrics, array_of_yearranges, array_of_binwidths):
        short_year_range = year_range[2:4] + "-" + year_range[7:]
        label = f"{short_year_range}"

        if len(metrics) < 4:
            metrics = metrics + [None] * (4 - len(metrics))

        for i, score in enumerate(score_names):
            if metrics[i] is not None:
                metrics_per_score[score].append((label, metrics[i], binwidth))

    unique_binwidths = sorted(set(array_of_binwidths))
    color_map = get_cmap("tab10")
    binwidth_colors = {bw: color_map(i) for i, bw in enumerate(unique_binwidths)}

    for score in score_names:
        entries = metrics_per_score[score]
        total_expected = len(array_of_metrics)
        total_actual = len(entries)

        if total_actual == 0:
            print(f"All values for '{score}' are None — skipping plot.")
            continue
        elif total_actual < total_expected:
            print(f"Some values for '{score}' are None ({total_expected - total_actual} missing).")

        plt.figure(figsize=(12, 6))
        
        x_labels = [e[0] for e in entries]
        y_vals = [e[1] for e in entries]
        binwidths = [e[2] for e in entries]
        x_pos = np.arange(len(x_labels))

        for i, (x, y, bw) in enumerate(zip(x_pos, y_vals, binwidths)):
            plt.scatter(x, y, color=binwidth_colors[bw], label=f"{bw} years" if x == x_pos[0] else "", s=60)

        plt.xlabel("Year Range")
        plt.ylabel(score)
        plt.title(f"{score} for Different Binning Widths")
        plt.xticks(x_pos, x_labels, rotation=45, fontsize=3)
        plt.grid()

        handles = []
        labels_seen = set()
        for bw in binwidths:
            if bw not in labels_seen:
                handles.append(plt.Line2D([], [], marker='o', color='w', label=f"{bw} years",
                                          markerfacecolor=binwidth_colors[bw], markersize=10))
                labels_seen.add(bw)
        plt.legend(handles=handles, title="Binning Width")

        plot_path = os.path.join(store_dir, f"{score.replace(' ', '_').lower()}_binnings.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\n Colored comparison plots saved in: {store_dir}")
