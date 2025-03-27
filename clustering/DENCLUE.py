import numpy as np
from clustering_utils import ClusteringResult  # Ensure correct import
from scipy.spatial.distance import cdist

def denclue_clustering(data: np.array, epsilon: float = 1e-5, max_iter: int = 100, bandwidth: float = 1.0):
    """
    Perform DENCLUE clustering.

    Args:
        data (np.array): The dataset to cluster.
        epsilon (float): Convergence tolerance for attractor density.
        max_iter (int): Maximum number of iterations to find attractors.
        bandwidth (float): Bandwidth for the kernel function.

    Returns:
        ClusteringResult: Named tuple with labels and cluster centers.
    """
    # Kernel function (Gaussian)
    def kernel_function(d, bandwidth):
        return np.exp(-0.5 * (d / bandwidth)**2)

    n_samples, n_features = data.shape
    labels = -1 * np.ones(n_samples)  # Initialize all labels as noise (-1)
    cluster_centers = []

    # Compute pairwise distances
    distances = cdist(data, data)
    
    # Initialize densities for each point
    densities = np.zeros(n_samples)
    for i in range(n_samples):
        densities[i] = np.sum(kernel_function(distances[i], bandwidth))

    # Iterate to find attractors (high-density regions)
    for _ in range(max_iter):
        new_labels = np.copy(labels)

        for i in range(n_samples):
            # Find the attractor (maximum density) closest to point i
            # Assign points to the cluster with the maximum density
            if densities[i] > epsilon:
                # Point i should belong to the attractor with the highest density
                nearest_attractor = np.argmax(densities)
                new_labels[i] = nearest_attractor

        # Check for convergence
        if np.all(new_labels == labels):
            break
        
        labels = np.copy(new_labels)

    # Find cluster centers
    unique_clusters = np.unique(labels[labels != -1])  # Exclude noise points (-1)
    
    cluster_centers = []
    for cluster in unique_clusters:
        # Get points in the cluster
        cluster_points = data[labels == cluster]
        # Compute the mean of the points in the cluster as the center
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)

    cluster_centers = np.array(cluster_centers)

    return ClusteringResult(labels=labels, cluster_centers=cluster_centers, data=data)
