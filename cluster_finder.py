#standard modules
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from collections import namedtuple
from numba import njit
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

#sklearn stuff (conda install scikit-learn)
from sklearn.cluster import MeanShift
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

#.py files
import getdata
import sortdata

#named tuple: https://docs.python.org/3/library/collections.html#collections.namedtuple
ClusterData = namedtuple("ClusterData", ["inc", "raan", "ecc"])
ClusteringResult = namedtuple("ClusteringResult", ["labels", "cluster_centers", "data"])

def adjust_raan_range(raan_values):
    # Convert RAAN from [0, 360] to [-180, 180]
    raan_adjusted = np.mod(np.array(raan_values) + 180, 360) - 180
    return raan_adjusted

def prepare_data_for_clustering(filename: str) -> ClusterData:
    data = getdata.array_extender(filename)
    data = np.array(data)
    data_TLE, data_frag, data_rest = sortdata.data_sorter(data, semi_major_index = 8, ecc_index = 10, inc_index = 9, mag_index = 20, source_index = 3)
    data = np.hstack([data_frag, data_rest])

    inc = data[9]
    raan = data[12]
    raan = adjust_raan_range(raan)
    ecc = data[10]
    return ClusterData(inc=inc, raan=raan, ecc=ecc)

#@njit
def find_clusters_mean_shift_clustering(data: ClusterData, bandwidth: float = 0.1) -> ClusteringResult:
    # Combine data
    combined_data = np.vstack((data.inc, data.raan, data.ecc)).T

    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(combined_data)

    # Perform MeanShift clustering
    mean_shift = MeanShift(bandwidth=bandwidth)
    mean_shift.fit(normalized_data)
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_

    # Unnormalize the data and cluster centers
    unnormalized_data = unnormalize(normalized_data, scaler.data_min_, scaler.data_max_)
    original_scale_centers = scaler.inverse_transform(cluster_centers)

    # Return the result as a namedtuple
    return ClusteringResult(labels=labels, cluster_centers=original_scale_centers,
                            data=unnormalized_data)

    
def compute_cost_function(data, bandwidth):
    """
    Compute the silhouette score for a given bandwidth value in MeanShift clustering.
    A higher silhouette score indicates better clustering.
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

def find_best_bandwidth(data, bandwidth_range):
    """
    Find the optimal bandwidth that maximizes the silhouette score.
    """
    best_bandwidth = None
    best_score = -1  # Start with a low score
    scores = []
    
    for b in bandwidth_range:
        score =compute_cost_function(data, b)
        scores.append(score)
        print(f"Bandwidth: {b}, Silhouette Score: {score}")
        if score > best_score:
            best_score = score
            best_bandwidth = b
    
    print(f"Best bandwidth: {best_bandwidth} with Silhouette Score: {best_score}")
    return best_bandwidth, scores

def plot_bandwidth_against_score(bandwidths: np.array, scores: np.array): 
    plt.figure(figsize=(8, 6))
    plt.plot(bandwidths, scores, marker='o', linestyle='-', color='b', label='Silhouette Score')
    plt.xlabel('Bandwidth')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Bandwidth')
    plt.grid(True)
    plt.xticks(bandwidths)
    plt.yticks(np.linspace(min(scores), max(scores), 5))
    plt.legend()
    plt.show()
    
def store_clusters(data, labels):
    """
    Store data points of each cluster into a dictionary of numpy arrays.
    
    Parameters:
        data (np.ndarray): The data points (NxM array).
        labels (np.ndarray): The cluster labels for each data point.
    
    Returns:
        clusters (dict): A dictionary where keys are cluster labels, and values are numpy arrays of cluster points.
    """
    clusters = {}
    unique_clusters = np.unique(labels)
    for cluster_label in unique_clusters:
        cluster_points = data[labels == cluster_label]
        clusters[cluster_label] = cluster_points
    return clusters

def add_large_cluster_info(data_small, labels_small, data_large, labels_large):
    """
    Add the corresponding large dataset cluster label as an additional column
    for points in the small dataset clusters.

    Parameters:
        data_small (np.ndarray): Data points in the small dataset (NxM array).
        labels_small (np.ndarray): Cluster labels for the small dataset.
        data_large (np.ndarray): Data points in the large dataset (PxM array).
        labels_large (np.ndarray): Cluster labels for the large dataset.
    
    Returns:
        updated_clusters_small (dict): A dictionary where each small cluster contains
                                       points with an additional column indicating 
                                       the large dataset cluster label.
    """
    # Create clusters for the small dataset
    clusters_small = store_clusters(data_small, labels_small)
    
    # Find the corresponding large cluster label for each point in the small dataset
    large_labels_for_small_data = []
    for point in data_small:
        # Find the index of the point in the large dataset
        idx = np.where((data_large == point).all(axis=1))[0][0]
        large_cluster_label = labels_large[idx]
        large_labels_for_small_data.append(large_cluster_label)
    
    # Convert to numpy array for easier slicing
    large_labels_for_small_data = np.array(large_labels_for_small_data)
    
    # Add the large cluster labels as a new column in each small cluster
    updated_clusters_small = {}
    for cluster_label, cluster_points in clusters_small.items():
        # Extract corresponding large labels for the current small cluster
        indices = np.where(labels_small == cluster_label)[0]
        large_labels = large_labels_for_small_data[indices]
        
        # Append the large cluster labels as a new column
        cluster_with_large_labels = np.hstack((cluster_points, large_labels[:, None]))
        updated_clusters_small[cluster_label] = cluster_with_large_labels
    
    return updated_clusters_small

def unnormalize(normalized_data, data_min=None, data_max=None):

    if data_min is None:
        data_min = np.array([0, -180, 0]) 
    if data_max is None:
        data_max = np.array([22, 180, 1])  
    return normalized_data * (data_max - data_min) + data_min

def convert_clusters_to_array(clusters_dict):
    """
    Convert cluster dictionary to a numpy array of the form:
    [cluster_label, inc, raan, ecc]
    """
    cluster_list = []
    for cluster_label, cluster_elements in clusters_dict.items():
        for element in cluster_elements:
            # Prepend the cluster label to each element
            cluster_list.append([cluster_label] + list(element))
    return np.array(cluster_list)

def is_subset(array_small, array_large, atol=1e-6):
    """
    Check if all rows in array_small exist in array_large within a tolerance.
    :param array_small: Smaller array (subset candidate)
    :param array_large: Larger array (superset candidate)
    :param atol: Absolute tolerance for numerical comparison
    :return: True if array_small is a subset of array_large, False otherwise
    """
    # Iterate over each row in the smaller array
    for row in array_small:
        matched = np.any(np.all(np.isclose(array_large, row, atol=atol), axis=1))
        if not matched:
            print(f"No match found for row: {row}")  # Debugging line
            return False
    return True

def get_files_for_cluster_evolution(year: str, orbit_type: str, seed: int, inp_directory: str): 
    year2 = year[2:]
    if orbit_type not in ["geo", "gto", "fol"]: 
        raise ValueError("Orbit type must be geo, gto, or fol!")
    if seed not in [1, 2, 3, 4]: 
        raise ValueError("Invalid seed value!")
    
    file_5mm = f"small_Master_{year2}_{orbit_type}_s{seed}.crs"
    file_10cm = f"stat_Master_{year2}_{orbit_type}_s{seed}.crs"
    
    file_5mm = os.path.join(inp_directory, file_5mm)
    file_10cm = os.path.join(inp_directory, file_10cm)

    return file_5mm, file_10cm

def find_clusters_for_one_year(file_5mm: str, file_10cm: str, bandwidths: list): 
    data_5mm = prepare_data_for_clustering(file_5mm)
    data_10cm = prepare_data_for_clustering(file_10cm)
    
    best_bandwidth_5mm, scores_5mm = find_best_bandwidth(data_5mm, bandwidths)
    clusters_5mm = find_clusters_mean_shift_clustering(data_5mm, best_bandwidth_5mm)
    cluster_data_5mm = clusters_5mm.data
    labels_5mm = clusters_5mm.labels
    cluster_centers_5mm = clusters_5mm.cluster_centers
    clusters_5mm_dict = store_clusters(cluster_data_5mm, labels_5mm)
    clusters_5mm_array = convert_clusters_to_array(clusters_5mm_dict)
    
    best_bandwidth_10cm, scores_10cm = find_best_bandwidth(data_10cm, bandwidths)
    clusters_10cm = find_clusters_mean_shift_clustering(data_10cm, best_bandwidth_10cm)
    cluster_data_10cm = clusters_10cm.data
    labels_10cm = clusters_10cm.labels
    cluster_centers_10cm = clusters_10cm.cluster_centers
    clusters_10cm_dict = store_clusters(cluster_data_10cm, labels_10cm)
    clusters_10cm_array = convert_clusters_to_array(clusters_10cm_dict)

    return clusters_5mm, clusters_10cm, clusters_5mm_array, clusters_10cm_array

def cluster_comparison(clusters_5mm_array: np.array, clusters_10cm_array: np.array): 
    count = 0  # Count number of matches
    updated_10cm_cluster_data = []  # Initialize an empty list to store updated cluster data

    data_10cm = clusters_10cm_array[:, 1:]  # Only the inc, raan, ecc; cluster label is removed
    data_5mm = clusters_5mm_array[:, 1:]

    for row_index, row_10cm in enumerate(data_10cm):  # Iterate over rows in the 10cm dataset
        matched = False
        matches = np.all(np.isclose(data_5mm, row_10cm, atol=1e-20), axis=1)  # Tolerance to 1e-20
        
        if np.any(matches):
            matched_indices = np.where(matches)[0]
            for idx in matched_indices:
                matched_row = clusters_5mm_array[idx]
                updated_10cm_cluster_data.append(
                    [clusters_10cm_array[row_index, 0]] + list(row_10cm) + [matched_row[0]]
                )
                matched = True
                break  # Stop searching once the first match is found

        if not matched:
            count += 1

    updated_10cm_cluster_data = np.array(updated_10cm_cluster_data)

    print("Updated 10cm Cluster Data with 5mm Cluster Labels:")
    print(updated_10cm_cluster_data)
    print(f"Number of unmatched 10cm elements: {count}")  # Should be zero

    first_column = updated_10cm_cluster_data[:, 0]  # First column: 10cm cluster labels
    last_column = updated_10cm_cluster_data[:, -1]  # Last column: 5mm cluster labels
    mismatches = first_column != last_column
    num_mismatches = np.sum(mismatches)
    print(f"Number of rows where the first and last columns do not match: {num_mismatches}")
    print(f"Total number of rows: {updated_10cm_cluster_data.shape[0]}")
    print(f"Mismatch percentage: {num_mismatches / updated_10cm_cluster_data.shape[0] * 100:.2f}%")
    
    return updated_10cm_cluster_data

def plot_cluster_center_evolution(cluster_centers_5mm_dict, cluster_centers_10cm_dict):
    orbit_types = set(key[1] for key in cluster_centers_5mm_dict.keys())

    for orbit_type in orbit_types:
        # Plot 5mm cluster centers
        fig_5mm = plt.figure(figsize=(12, 8))
        ax_5mm = fig_5mm.add_subplot(111, projection='3d')

        for (year, orbit, seed), centers_5mm in cluster_centers_5mm_dict.items():
            if orbit != orbit_type:
                continue

            # Plot 5mm cluster centers
            ax_5mm.scatter(
                centers_5mm[:, 0], centers_5mm[:, 1], centers_5mm[:, 2],
                label=f"{year} (5mm)", alpha=0.7, marker='o'
            )

            # Annotate 5mm cluster centers
            for i, center in enumerate(centers_5mm):
                ax_5mm.text(center[0], center[1], center[2], f"{year}", size=8)

        ax_5mm.set_xlabel('Inclination')
        ax_5mm.set_ylabel('RAAN')
        ax_5mm.set_zlabel('Eccentricity')
        ax_5mm.set_title(f'5mm Cluster Center Evolution for Orbit Type: {orbit_type.upper()}')
        ax_5mm.legend()
        plt.show()

        # Plot 10cm cluster centers
        fig_10cm = plt.figure(figsize=(12, 8))
        ax_10cm = fig_10cm.add_subplot(111, projection='3d')

        for (year, orbit, seed), centers_10cm in cluster_centers_10cm_dict.items():
            if orbit != orbit_type:
                continue

            # Plot 10cm cluster centers
            ax_10cm.scatter(
                centers_10cm[:, 0], centers_10cm[:, 1], centers_10cm[:, 2],
                label=f"{year} (10cm)", alpha=0.7, marker='^'
            )

            # Annotate 10cm cluster centers
            for i, center in enumerate(centers_10cm):
                ax_10cm.text(center[0], center[1], center[2], f"{year}", size=8)

        ax_10cm.set_xlabel('Inclination i')
        ax_10cm.set_ylabel('RAAN')
        ax_10cm.set_zlabel('Eccentricity')
        ax_10cm.set_title(f'10cm Cluster Center Evolution for Orbit Type: {orbit_type.upper()}')
        ax_10cm.legend()
        plt.show()
        
def plot_cluster_center_evolution_2d(cluster_centers_5mm_dict, cluster_centers_10cm_dict):
    orbit_types = set(key[1] for key in cluster_centers_5mm_dict.keys())

    for orbit_type in orbit_types:
        # Plot 5mm cluster centers in 2D
        fig_5mm = plt.figure(figsize=(8, 6))
        ax_5mm = fig_5mm.add_subplot(111)

        for (year, orbit, seed), centers_5mm in cluster_centers_5mm_dict.items():
            if orbit != orbit_type:
                continue

            # Plot 5mm cluster centers in 2D
            ax_5mm.scatter(
                centers_5mm[:, 0], centers_5mm[:, 1],
                label=f"{year} (5mm)", alpha=0.7, marker='o'
            )

            # Annotate 5mm cluster centers
            for i, center in enumerate(centers_5mm):
                ax_5mm.text(center[0], center[1], f"{year}", size=8)

        ax_5mm.set_xlabel('Inclination')
        ax_5mm.set_ylabel('RAAN')
        ax_5mm.set_title(f'5mm Cluster Center Evolution for Orbit Type: {orbit_type.upper()}')
        ax_5mm.legend()
        plt.grid(True)
        plt.show()

        # Plot 10cm cluster centers in 2D
        fig_10cm = plt.figure(figsize=(12, 8))
        ax_10cm = fig_10cm.add_subplot(111)

        for (year, orbit, seed), centers_10cm in cluster_centers_10cm_dict.items():
            if orbit != orbit_type:
                continue

            # Plot 10cm cluster centers in 2D
            ax_10cm.scatter(
                centers_10cm[:, 0], centers_10cm[:, 1],
                label=f"{year} (10cm)", alpha=0.7, marker='^'
            )

            # Annotate 10cm cluster centers
            for i, center in enumerate(centers_10cm):
                ax_10cm.text(center[0], center[1], f"{year}", size=8)

        ax_10cm.set_xlabel('Inclination i')
        ax_10cm.set_ylabel('RAAN')
        ax_10cm.set_title(f'10cm Cluster Center Evolution for Orbit Type: {orbit_type.upper()}')
        ax_10cm.legend()
        plt.grid(True)
        plt.show()

def match_clusters(cluster_centers_10cm: np.array, cluster_centers_5mm: np.array):
    """
    Hungarian algorithm to match the cluster centers by calculating pairwise distances between the 
    clusters and filling them in a matrix. The Hungarian algorithm tries to find the permutation of the
    matrix rows that minimizes the trace. 
    https://en.wikipedia.org/wiki/Hungarian_algorithm and DOI: 10.1109/TAES.2016.140952
    
    This function is an adapted version of the Hungarian algorithm for matrices that are not square.
    
    Args:
        cluster_centers_10cm (np.array): Cluster centers of the 10 cm dataset (shape: n x d).
        cluster_centers_5mm (np.array): Cluster centers of the 5 mm dataset (shape: m x d).

    Returns:
        matches (np.array): Matched pairs and their distances, excluding dummy matches.
        unmatched_10cm (list): Indices of unmatched clusters in the 10cm dataset.
        unmatched_5mm (list): Indices of unmatched clusters in the 5mm dataset.
    """
    distance_matrix = cdist(cluster_centers_10cm, cluster_centers_5mm, metric='euclidean')
    n, m = distance_matrix.shape

    if n > m:
        padded_matrix = np.hstack((distance_matrix, np.full((n, n - m), 10_000)))
        #I cannot set the dummy matches to infinity because the linear_sum_assignment does not accept Inf or Naan
        #Setting the distance to 10_000 makes it very costly, so these dummy matches will be ignored
    elif m > n:
        padded_matrix = np.vstack((distance_matrix, np.full((m - n, m), 10_000)))
    else:
        padded_matrix = distance_matrix

    row_indices, col_indices = linear_sum_assignment(padded_matrix)

    # Identify matches
    valid_matches = []
    unmatched_10cm = []
    unmatched_5mm = []

    for row, col in zip(row_indices, col_indices):
        if row < n and col < m:
            valid_matches.append((row, col, distance_matrix[row, col]))
        elif row < n and col >= m:
            unmatched_10cm.append(row)  # Unmatched 10cm clusters
        elif row >= n and col < m:
            unmatched_5mm.append(col)  # Unmatched 5mm clusters

    matches = np.array(valid_matches)
    return matches, unmatched_10cm, unmatched_5mm
