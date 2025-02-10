import numpy as np
from sklearn.cluster import DBSCAN
from cluster_data import estimate_runtime
from cluster_plotter import ClusterPlotter

def dbscan_clustering(data: np.array, eps: float = 0.5, min_samples: int = 5):
    """
    Perform DBSCAN clustering.

    Args:
        data (np.array): The dataset to cluster.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        labels (np.array): Cluster labels for each point.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels
