import numpy as np
from sklearn.cluster import DBSCAN
from clustering_utils import ClusteringResult  # Ensure correct import

def dbscan_clustering(data: np.array, eps: float = 10, min_samples: int = 10):
    """
    Perform DBSCAN clustering.

    Args:
        data (np.array): The dataset to cluster.
        eps (float): Maximum distance between two samples for them to be considered neighbors / neighbourhood radius.
        min_samples (int): Number of points required to form a dense region.

    DBSCAN categorizes points into three types based on their density:  
    1. **Core Points**:  
    - A point is a **core point** if at least **min_samples** points (including itself) exist within its $\epsilon$-radius.  
    - Core points form the foundation of dense clusters.  
    2. **Border Points**:  
    - A **border point** has fewer than **min_samples** neighbors within $\epsilon$, but it is reachable from a core point.  
    - These points are part of a cluster but do not contribute to forming new clusters.  
    3. **Noise (Outlier) Points**:  
    - A **noise point** (or outlier) is neither a core point nor a border point.  
    - It does not belong to any cluster and is considered an anomaly.  

    - Clusters start from **core points**, expanding outward by including directly and indirectly connected core points.  
    - **Border points** attach to the nearest core point but do not expand the cluster.  
    - **Noise points** are ignored and remain unclustered.  

    Density-Reachable: A point $q$ is density-reachable from $p$ if $p$ is a core point and $q$ is within its $\epsilon$-radius. This can extend through a chain of core points.
    Density-Connected: Two points $p$ and $q$ are density-connected if there is a core point $o$ linking them via density-reachable paths.

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

