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
        ClusteringResult: Named tuple with labels and dummy cluster centers.
    """
    # Kernel function (Gaussian)
    def kernel_function(d, bandwidth):
        return np.exp(-0.5 * (d / bandwidth)**2)

    n_samples, n_features = data.shape
    labels = -1 * np.ones(n_samples)
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
            nearest_attractor = np.argmax(densities[i])

            if densities[i] > epsilon:
                new_labels[i] = nearest_attractor

        # Check for convergence
        if np.all(new_labels == labels):
            break
        
        labels = np.copy(new_labels)

    # Find cluster centers
    unique_clusters = np.unique(labels[labels != -1])  # Exclude noise points (-1)
    cluster_centers = np.array([np.mean(data[labels == cluster], axis=0) if np.any(labels == cluster) else np.nan
                                for cluster in unique_clusters])

    return ClusteringResult(labels=labels, cluster_centers=cluster_centers, data=data)