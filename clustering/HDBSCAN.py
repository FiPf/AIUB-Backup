#https://scikit-learn.org/stable/modules/clustering.html#hdbscan
#Campello, R.J.G.B., Moulavi, D., Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. In: Pei, J., Tseng, V.S., Cao, L., Motoda, H., Xu, G. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2013. Lecture Notes in Computer Science(), vol 7819. Springer, Berlin, Heidelberg. Density-Based Clustering Based on Hierarchical Density Estimates
#L. McInnes and J. Healy, (2017). Accelerated Hierarchical Density Based Clustering. In: IEEE International Conference on Data Mining Workshops (ICDMW), 2017, pp. 33-42. Accelerated Hierarchical Density Based Clustering

import numpy as np
import hdbscan
from clustering_utils import ClusteringResult  # Ensure correct import

def hdbscan_clustering(data: np.array, min_cluster_size: int = 10, min_samples: int = None, cluster_selection_epsilon: float = 0.0, plot_condensed_tree: bool = False):
    """
    Perform HDBSCAN clustering.

    Args:
        data (np.array): The dataset to cluster.
        min_cluster_size (int): Minimum number of points required to form a cluster.
        min_samples (int or None): Number of samples in a neighborhood for a point to be a core point. If None, defaults to min_cluster_size.
        cluster_selection_epsilon (float): A distance threshold for splitting clusters (optional, typically 0).
        plot_condensed_tree (bool): If True, displays the condensed tree plot.

    HDBSCAN operates by:
    - Building a hierarchical tree of clusters based on density.
    - Condensing this tree to find stable clusters at different density levels.
    - Assigning labels based on cluster stability, leaving some points as noise

    Unlike DBSCAN:
    - HDBSCAN does not require a fixed epsilon.
    - It can find clusters with variable densities.
    - It returns soft clustering probabilities, indicating how strongly a point belongs to a cluster.

    Returns:
        ClusteringResult: Named tuple with labels and cluster centers (calculated with mean).
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
    labels = clusterer.fit_predict(data)

    unique_clusters = np.unique(labels[labels != -1])  # Ignore noise (-1)
    cluster_centers = np.array([np.mean(data[labels == cluster], axis=0) if np.any(labels == cluster) else np.nan
                                for cluster in unique_clusters])

    # Plot condensed tree if requested
    if plot_condensed_tree:
        clusterer.condensed_tree_.plot()
    
    return ClusteringResult(labels=labels, cluster_centers=cluster_centers, data=data)
