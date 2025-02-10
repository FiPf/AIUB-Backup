from collections import namedtuple
import numpy as np
import time 
import my_kmedoids
from clustering_utils import ClusteringResult
from sklearn.preprocessing import MinMaxScaler

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import getdata
import sortdata

#change the ClusterData namedtuple if you want to add more dimensions
ClusterData = namedtuple("ClusterData", ["inc", "raan", "ecc"])

def cluster_data_to_array(data_list: namedtuple):
    """Convert ClusterData namedtuple to an numpy array. Works for any number of dimensions.

    Args:
        data_list (namedtuple): data to be converted

    Returns:
        np.array: converted data
    """    
    return np.array([tuple(item) for item in data_list])

def normalize_data(arr: np.array):
    """Normalize data to zero mean and unit variance. Important for clustering. 

    Args:
        arr (np.array): Input data array.

    Returns:
        normalized (np.array): Normalized data.
        data_min (np.array): Minimum values before normalization.
        data_max (np.array): Maximum values before normalization.
    """    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(arr)
    return normalized_data, scaler.data_min_, scaler.data_max_

def unnormalize(normalized_data: np.array, cluster_centers: np.array, data_min: np.array, data_max: np.array): 
    """Reverts the normalization process for both data and cluster centers.

    Args:
        normalized_data (np.array): Normalized dataset.
        cluster_centers (np.array): Normalized cluster centers.
        data_min (np.array): Minimum values used for normalization.
        data_max (np.array): Maximum values used for normalization.

    Returns:
        np.array: Unnormalized data.
        np.array: Unnormalized cluster centers.
    """
    unnormalized_data = normalized_data * (data_max - data_min) + data_min
    unnormalized_centers = cluster_centers * (data_max - data_min) + data_min
    return unnormalized_data, unnormalized_centers

def adjust_raan_range(raan_values):
    """Convert RAAN from [0, 360]° to [-180, 180]°

    Args:
        raan_values (np.array / list): list of raan values between 0 to 360 degrees

    Returns:
        np.array / list: list of raan values between -180 to 180 degrees
    """    
    raan_adjusted = np.mod(np.array(raan_values) + 180, 360) - 180
    return raan_adjusted

def prepare_data_for_clustering(filename: str) -> ClusterData:
    """get data from *.crs or *.det file with array_extender, do sorting, adjust the raan range. 

    Args:
        filename (str): *.crs or *.det file to get data from

    Returns:
        ClusterData: data ready for clustering, contains inc, raan and ecc arrays
    """
    data = getdata.array_extender(filename)
    data = np.array(data)
    data_TLE, data_frag, data_rest = sortdata.data_sorter(data, semi_major_index = 8, ecc_index = 10, inc_index = 9, mag_index = 20, source_index = 3)
    data = np.hstack([data_frag, data_rest])

    inc = data[9]
    raan = data[12]
    raan = adjust_raan_range(raan)
    ecc = data[10]
    return ClusterData(inc=inc, raan=raan, ecc=ecc)

def estimate_runtime(clustering_func, *args, build_function=None, swap_function=None, **kwargs):
    """
    Measures the execution time of a clustering function. If build_function or swap_function
    is provided, uses pam_clustering instead of clustering_func.

    Args:
        clustering_func (Callable): The clustering function to evaluate.
        *args: Positional arguments for the clustering function (expects data and k).
        build_function (Callable, optional): The BUILD function for pam_clustering. Defaults to None.
        swap_function (Callable, optional): The SWAP function for pam_clustering. Defaults to None.
        **kwargs: Additional keyword arguments for the clustering function.

    Returns:
        Tuple (result, runtime_in_seconds)
    """
    #print("args:", args)
    #print("kwargs:", kwargs)
    start_time = time.time()

    # Extract data and k properly
    if len(args) < 2:
        raise ValueError("Expected at least two positional arguments: (data, k)")
    
    data, k = args[0], args[1]

    if not isinstance(k, int):
        raise TypeError(f"Expected k to be an integer, but got {type(k).__name__}: {k}")

    # Ensure pam_clustering uses the correct build and swap functions
    if clustering_func == my_kmedoids.pam_clustering:
        build_function = build_function or my_kmedoids.pam_build
        swap_function = swap_function or my_kmedoids.pam_swap
        result = my_kmedoids.pam_clustering(data, k, build_function, swap_function)#problem here
    else:
        result = clustering_func(data, k, **kwargs)

    end_time = time.time()
    runtime = end_time - start_time

    print(f"Runtime for {clustering_func.__name__ if clustering_func != my_kmedoids.pam_clustering else 'pam_clustering'}"
          f" (build: {build_function.__name__ if build_function else 'default build'}, "
          f"swap: {swap_function.__name__ if swap_function else 'default swap'}): {runtime:.6f} seconds")

    return result, runtime