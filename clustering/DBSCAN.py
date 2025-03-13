import numpy as np
from sklearn.cluster import DBSCAN
from clustering_utils import ClusteringResult  # Ensure correct import

def dbscan_clustering(data: np.array, eps: float = 10, min_samples: int = 10):
    """
    Perform DBSCAN clustering.

    Args:
        data (np.array): The dataset to cluster.
        eps (float): Maximum distance between two samples for them to be considered neighbors.
        min_samples (int): Number of points to form a dense region.

    Returns:
        ClusteringResult: Named tuple with labels and dummy cluster centers.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)

    # DBSCAN does not have cluster centers, so we use NaN placeholders
    unique_clusters = np.unique(labels[labels != -1])  # Ignore noise (-1)
    cluster_centers = np.array([np.mean(data[labels == cluster], axis=0) if np.any(labels == cluster) else np.nan
                                for cluster in unique_clusters])
    # we return None as a placeholder for cluster centers to make it compatible with the other algorithms

    
    return ClusteringResult(labels=labels, cluster_centers=cluster_centers, data=data)

