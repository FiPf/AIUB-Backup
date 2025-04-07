import numpy as np
from sklearn.cluster import OPTICS
from clustering_utils import ClusteringResult
import matplotlib.pyplot as plt

def optics_clustering(data: np.array, min_samples, max_eps, xi, plot_reachability: bool = False):
    """
    Perform OPTICS clustering.

    Args:
        data (np.array): The dataset to cluster.
        min_samples (int): Minimum number of samples in a neighborhood to form a core point.
        max_eps (float): Maximum distance between samples for connectivity (default: infinity for full reachability).
        xi (float): Determines the minimum steepness on the reachability plot to define clusters. xi large results in few, but larger 
        clusters, xi small gives more detailed clusters.

    OPTICS (Ordering Points To Identify Clustering Structure) works similarly to DBSCAN but provides a reachability plot
    and allows extraction of clusters at different density thresholds:
    
    - Core Points: A point is a core point if at least min_samples points exist within its neighborhood.
    - Reachability Distance: Measures how far a point is from its nearest core point.
    - Cluster Extraction: Clusters are extracted from the reachability plot based on density variations.
    - Unlike DBSCAN, OPTICS does not require a fixed `eps` value but dynamically detects clusters of varying densities.

    https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html#sphx-glr-auto-examples-cluster-plot-optics-py
    https://en.wikipedia.org/wiki/OPTICS_algorithm

    Returns:
        ClusteringResult: Named tuple with labels and cluster centers (calculated with mean).
    """
    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=xi)
    labels = optics.fit_predict(data)
    
    unique_clusters = np.unique(labels[labels != -1])  # Ignore noise (-1)
    cluster_centers = np.array([
        np.mean(data[labels == cluster], axis=0) if np.any(labels == cluster) else np.nan
        for cluster in unique_clusters
    ])

    reachability = optics.reachability_
    ordering = optics.ordering_

    if plot_reachability:
        plt.figure(figsize=(10, 5))
        plt.scatter(ordering, reachability[ordering], marker='o', linestyle='-',s =2)
        plt.xlabel("Data Points (Ordered)")
        plt.ylabel("Reachability Distance")
        plt.title("OPTICS Reachability Plot")
        plt.show()
    
    return ClusteringResult(labels=labels, cluster_centers=cluster_centers, data=data)