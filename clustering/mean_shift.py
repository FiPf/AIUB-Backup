import numpy as np
import matplotlib.pyplot as plt
import os

#sklearn stuff (conda install scikit-learn)
from sklearn.cluster import MeanShift
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import getdata
from getdata import PopulationType
import sortdata

from cluster_data import ClusteringResult, ClusterData
from clustering_utils import ClusteringResult
import cluster_data

def mean_shift_clustering(data: ClusterData, bandwidth: float = 0.1) -> ClusteringResult:
    """Performs Mean Shift Clustering on orbital data (inc, raan, ecc). The function normalizes the input data using Min-Max scaling before clustering 
    and reverts the results back to the original scale for interpretability.

    Args:
        data (ClusterData): A named tuple containing the orbital elements: inc, raan, ecc. 
        bandwidth (float, optional): The bandwidth parameter for the Mean Shift algorithm, controlling the window size 
        for clustering. Smaller values detect finer clusters, while larger values merge 
        nearby points. Defaults to 0.1.

    Returns:
        ClusteringResult: a named tuple containing: 
        - `labels`: Cluster labels assigned to each data point.
        - `cluster_centers`: Cluster centers in the original (unnormalized) scale.
        - `data`: The unnormalized data after inverse scaling.
    """    
    # Combine data
    combined_data = np.vstack((data.inc, data.raan )).T#, data.ecc)).T

    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(combined_data)

    # Perform MeanShift clustering
    mean_shift = MeanShift(bandwidth=bandwidth)
    mean_shift.fit(normalized_data)
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_

    # Unnormalize the data and cluster centers
    unnormalized_data = cluster_data.unnormalize(normalized_data, scaler.data_min_, scaler.data_max_)
    original_scale_centers = scaler.inverse_transform(cluster_centers)

    # Return the result as a namedtuple
    return ClusteringResult(labels=labels, cluster_centers=original_scale_centers,
                            data=unnormalized_data)

    
def compute_cost_function(data, bandwidth):
    """Compute the silhouette score for a given bandwidth value in MeanShift clustering.
    A higher silhouette score indicates better clustering.

    Args:
        data (named tuple): data to compute the cost for
        bandwidth (float): bandwidth tuning parameter for mean shift clustering

    Returns:
        score (float): silhouette score for the dataset with that given bandwidth 
    """    
    # Perform clustering using the given bandwidth
    cluster_data = find_clusters_mean_shift_clustering(data, bandwidth)
    labels = cluster_data.labels
    unnormalized_data = cluster_data.data
    
    # Calculate silhouette score (a higher value means better clustering)
    if len(np.unique(labels)) > 1:  # Ensure there is more than one cluster
        score = silhouette_score(unnormalized_data, labels)
    else:
        score = -1  # Return a negative score if clustering fails
        
    num_clusters = len(np.unique(labels))
    if num_clusters < 5: # at least 5 clusters
        score -= (5 - num_clusters) * 0.1  # Penalty for too few clusters
        
    return score