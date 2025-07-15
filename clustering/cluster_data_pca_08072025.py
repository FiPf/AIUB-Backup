from collections import namedtuple
import numpy as np
import time 
import my_kmedoids
import kmeans
import fuzzy_c_means
from clustering_utils_pca_08072025 import ClusteringResult, ClusterData
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import getdata
from getdata import PopulationType
import sortdata
from cluster_plotter import ClusterPlotter
import scores
import mean_shift, DBSCAN
from typing import Callable
import OPTICS
import HDBSCAN
import DENCLUE

def run_clustering(algorithm: Callable, name: str, data: np.array, data_min: np.array, data_max: np.array, *args, **kwargs):
    """
    Runs a given clustering algorithm. Meaning the algorithm is started, different scores are calculated, and the clustered data 
    is plotted if plotting is enabled. 

    Args:
        algorithm (Callable): function which starts the clustering algorithm
        name (str): name of the clustering algorithm
        data (np.array): data to be clustered
        data_min (np.array): tuple, contains the minimum of the data for each dimension
        data_max (np.array): tuple, contains the maximum of the data for each dimension

    Returns:
        result (ClusteringResult): named tuple containing labels, cluster_centers, and the data
        runtime (float): runtime of the algorithm
        n_clusters (int): number of clusters
        points_per_cluster (dict): number of points for each cluster, stored in a dictionary
        metrics (np.array): contains all the scores and metrics calculated for this algorithm
    """

    metrics = []  # List containing all the scores, 2D standard deviation, cluster densities
    metric = kwargs.pop("metric", False)  # Default to False if not provided

    # Start the clustering algorithm and calculate runtime
    result, runtime = estimate_runtime(algorithm, data, *args, **kwargs)

    n_clusters = len(set(result.labels))  # Number of unique labels (clusters)
    points_per_cluster = {i: list(result.labels).count(i) for i in set(result.labels)}  # Count points per cluster

    if n_clusters > 1 and metric:
        DB_sc = scores.DB_score(result)
        CH_sc = scores.CH_score(result)
        dunn_index_sc = scores.dunn_index_score(result)
        sil_sc = scores.sil_score(result)
        cluster_std = scores.cluster_std_eigen(result)
        #commented out, only works for two dimensions
        #square_density, square_bounds = scores.cluster_density_squares(result)
        #hull_density, hull_bounds = scores.cluster_density_convex_hull(result)
        metrics = [DB_sc, CH_sc, dunn_index_sc, sil_sc, cluster_std] #, square_density, hull_density]

    unnormalized_data, cluster_centers = unnormalize(result.data, result.cluster_centers, data_min, data_max)
    
    return result, runtime, n_clusters, points_per_cluster, metrics

def run_clustering_dbcv_score(algorithm: Callable, name: str, data: np.array, data_min: np.array, data_max: np.array, *args, **kwargs):
    """
    Runs a given clustering algorithm. Meaning the algorithm is started, different scores are calculated, and the clustered data 
    is plotted if plotting is enabled. 

    Args:
        algorithm (Callable): function which starts the clustering algorithm
        name (str): name of the clustering algorithm
        data (np.array): data to be clustered
        data_min (np.array): tuple, contains the minimum of the data for each dimension
        data_max (np.array): tuple, contains the maximum of the data for each dimension

    Returns:
        result (ClusteringResult): named tuple containing labels, cluster_centers, and the data
        runtime (float): runtime of the algorithm
        n_clusters (int): number of clusters
        points_per_cluster (dict): number of points for each cluster, stored in a dictionary
        metrics (np.array): contains all the scores and metrics calculated for this algorithm
    """

    metrics = []  # List containing all the scores, 2D standard deviation, cluster densities
    plot = kwargs.pop("plot", True)  # Default to True if not provided

    # Start the clustering algorithm and calculate runtime
    result, runtime = estimate_runtime(algorithm, data, *args, **kwargs)

    n_clusters = len(set(result.labels))  # Number of unique labels (clusters)
    points_per_cluster = {i: list(result.labels).count(i) for i in set(result.labels)}  # Count points per cluster

    if n_clusters > 1:
        print("Calculating DBCV score...")
        dbcv_score = scores.DBCV_score_rust(result)
        cluster_std = scores.cluster_std_eigen(result)
        square_density, square_bounds = scores.cluster_density_squares(result)
        hull_density, hull_bounds = scores.cluster_density_convex_hull(result)
        metrics = [dbcv_score, cluster_std, square_density, hull_density]

    unnormalized_data, cluster_centers = unnormalize(result.data, result.cluster_centers, data_min, data_max)
    
    # Plotting the clusters if enabled
    if plot: 
        plotter = ClusterPlotter(unnormalized_data, result.labels, cluster_centers)
        plotter.clusters_2d_plot(f"{name} - 2D Cluster Visualization")

    return result, runtime, n_clusters, points_per_cluster, metrics

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

def unnormalize(normalized_data: np.array, cluster_centers: np.array, data_min: np.array, data_max: np.array, labels=None): 
    """Reverts the normalization process for both data and cluster centers.
       If cluster_centers is None or an empty array, computes the cluster centers as the mean of data points assigned to each cluster.

    Args:
        normalized_data (np.array): Normalized dataset.
        cluster_centers (np.array): Normalized cluster centers.
        data_min (np.array): Minimum values used for normalization.
        data_max (np.array): Maximum values used for normalization.
        labels (np.array, optional): Labels assigned to the data points for clustering.

    Returns:
        np.array: Unnormalized data.
        np.array: Unnormalized cluster centers (or computed centers if `cluster_centers` is None or empty).
    """
    unnormalized_centers = None
    unnormalized_data = normalized_data * (data_max - data_min) + data_min

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
        ClusterData: data ready for clustering, contains inc, raan and perigee arrays
    """
    data = getdata.array_extender(filename)
    data = np.array(data)
    data_TLE, data_frag, data_rest = sortdata.data_sorter(data, semi_major_index = 8, ecc_index = 10, inc_index = 9, mag_index = 20, source_index = 3)
    data = np.hstack([data_frag, data_rest])

    inc = data[9]
    raan = data[12]
    raan = adjust_raan_range(raan)
    ecc = data[10]
    perigee = data[11]
    mag_obj = data[20]
    sem_maj = data[8]
    diameter = data[1]
    true_lat = data[13]
    mu = 3.986004418e14
    mean_motion = np.sqrt(mu/(sem_maj*1000)**3) #convert semi major from km to m
    mean_motion = mean_motion/(2*np.pi)*86_400
    return ClusterData(ecc=ecc, sem_maj=sem_maj, inc=inc, raan=raan, perigee=perigee, true_lat=true_lat, mean_motion=mean_motion, mag_obj=mag_obj, diameter=diameter)

def generate_running_year_ranges(start_year: int, end_year: int, window_size: int=4):
    """create year ranges to bin the data. The overlap of the year ranges is always window_size - 1. 
    Used to analyze the cluster evolution.

    Args:
        start_year (int): start of the data
        end_year (int): end of the data
        window_size (int, optional): How many years to put in one plot. Defaults to 4.

    Returns:
        year_ranges (np.array): years grouped in the desired way 
    """
    year_ranges = {}
    current_start = start_year
    while current_start + window_size - 1 <= end_year:
        current_end = current_start + window_size - 1
        year_ranges[f"{current_start}-{current_end}"] = np.arange(current_start, current_end + 1)
        current_start += 1
    return year_ranges

def merge_cluster_data(cluster_data_list: list):
    """Merge multiple ClusterData objects into a single ClusterData.

    Args:
        cluster_data_list (list): list with data to cluster

    Returns:
        ClusterData (named tuple): contains data with all dimensions 
    """

    sem_maj_all = np.concatenate([cd.sem_maj for cd in cluster_data_list])
    inc_all = np.concatenate([cd.inc for cd in cluster_data_list])
    ecc_all = np.concatenate([cd.ecc for cd in cluster_data_list])
    raan_all = np.concatenate([cd.raan for cd in cluster_data_list])
    perigee_all = np.concatenate([cd.perigee for cd in cluster_data_list])
    true_lat_all = np.concatenate([cd.true_lat for cd in cluster_data_list])
    mean_motion_all = np.concatenate([cd.mean_motion for cd in cluster_data_list])
    mag_obj_all = np.concatenate([cd.mag_obj for cd in cluster_data_list])
    diameter_all = np.concatenate([cd.diameter for cd in cluster_data_list])

    return ClusterData(ecc=ecc_all, sem_maj=sem_maj_all, inc=inc_all, raan=raan_all, perigee=perigee_all, true_lat=true_lat_all, mean_motion=mean_motion_all, mag_obj=mag_obj_all, diameter=diameter_all)

def bin_data_for_clustering(year_ranges: dict, print_res: bool = True):
    """Find the right files and bin the data according to the given year_ranges.

    Args:
        year_ranges (dict): grouped years how the data should be binned
        print_res (bool, optional): Print the filenames and year_ranges for a check. Defaults to True.

    Returns:
        results (list): List of (ClusterData, year_range) tuples
    """
    seeds = [1]
    orbit_types = ["geo", "gto", "fol"]
    file_lists = {year_range: {orbit: [] for orbit in orbit_types} for year_range in year_ranges}

    for year_range, years in year_ranges.items():
        for year in years:
            year_str = str(year)[2:]
            for orbit in orbit_types:
                for seed in seeds:
                    if year_str != "23":
                        file = f"../input/stat_Master_{year_str}_{orbit}_s{seed}.crs"
                    else:
                        file = f"../input/stat_Master_{year_str}_{orbit}_s{seed}_10cm.crs"
                    file_lists[year_range][orbit].append(file)

    if print_res:
        for year_range, orbits in file_lists.items():
            print(f"\nYear Range: {year_range}")
            for orbit, files in orbits.items():
                print(f"  {orbit.upper()} Files: {files}")

    results = []

    for year_range, orbits in file_lists.items():
        diameter_all, sem_maj_all, inc_all, ecc_all = [], [], [], []
        raan_all, mag_obj_all = [], []
        perigee_all, true_lat_all, mean_motion_all = [], [], []

        for orbit, files in orbits.items():
            cluster_data_list = [prepare_data_for_clustering(f) for f in files]
            merged_data = merge_cluster_data(cluster_data_list)

            diameter_all.append(merged_data.diameter)
            sem_maj_all.append(merged_data.sem_maj)
            inc_all.append(merged_data.inc)
            ecc_all.append(merged_data.ecc)
            raan_all.append(merged_data.raan)
            mag_obj_all.append(merged_data.mag_obj)
            perigee_all.append(merged_data.perigee)
            true_lat_all.append(merged_data.true_lat)
            mean_motion_all.append(merged_data.mean_motion)

        diameter_all = np.concatenate(diameter_all) if diameter_all else np.array([])
        sem_maj_all = np.concatenate(sem_maj_all) if sem_maj_all else np.array([])
        inc_all = np.concatenate(inc_all) if inc_all else np.array([])
        ecc_all = np.concatenate(ecc_all) if ecc_all else np.array([])
        raan_all = np.concatenate(raan_all) if raan_all else np.array([])
        mag_obj_all = np.concatenate(mag_obj_all) if mag_obj_all else np.array([])
        mean_motion_all = np.concatenate(mean_motion_all) if mean_motion_all else np.array([])
        perigee_all = np.concatenate(perigee_all) if perigee_all else np.array([])
        true_lat_all = np.concatenate(true_lat_all) if true_lat_all else np.array([])

        results.append((
        ClusterData(ecc=ecc_all, sem_maj=sem_maj_all, inc=inc_all, raan=raan_all, perigee=perigee_all, true_lat=true_lat_all, mean_motion=mean_motion_all, mag_obj=mag_obj_all, diameter=diameter_all), 
            year_range
        ))

    return results

#def bin_observed_data(uncorr_obs_files: list, year_ranges: dict, print_res: bool = False):
    results = []

    for year_range, years in year_ranges.items():
        obs_files = [
            os.path.join("..", "input", uncorr_obs_files[y])
            for y in years
            if y in uncorr_obs_files
        ]

        if print_res:
            print(f"\nYear Range: {year_range}")
            print(f"  Files: {obs_files}")

        data = getdata.array_extender_obs(obs_files)

        inc     = np.array(data[11], dtype=float)  
        raan    = np.array(data[12], dtype=float)  
        perigee = np.array(data[13],  dtype=float)  
        ecc     = np.array(data[9],  dtype=float)  
        mag     = np.array(data[20], dtype=float)  

        max_inc = 22
        valid_idx = np.where(inc < max_inc)[0]
        inc     = inc[valid_idx]
        inc[inc < 0] = 0
        raan    = raan[valid_idx]
        peigee = perigee[valid_idx]
        ecc     = ecc[valid_idx]
        mag     = mag[valid_idx]

        sorted_idx = np.argsort(raan)
        inc     = inc[sorted_idx]
        raan    = raan[sorted_idx]
        perigee = perigee[sorted_idx]
        ecc     = ecc[sorted_idx]
        mag     = mag[sorted_idx]

        raan = adjust_raan_range(raan)

        results.append((
            ClusterData(
                inc=inc,
                raan=raan,
                perigee=perigee,
                ecc=ecc,
                mag=mag
            ),
            year_range
        ))

    return results

def estimate_runtime(clustering_func: Callable, *args, build_function: Callable=None, swap_function: Callable=None, **kwargs):
    """Measures execution time of a clustering function, handling different clustering algorithms.

    Args:
        clustering_func (Callable): function which contains the clustering algorithm
        build_function (Callable, optional): Build function, only used for versions of PAM/kmedoids. Defaults to None.
        swap_function (Callable, optional): Swap function, only used for versions of PAM/kmedoids. Defaults to None.

    Returns:
        result (ClusteringResult): named tuple containing labels, cluster_centers and data
        runtime (float): runtime of the given algorithm
    """    
    start_time = time.time()
    
    # Identify the type of clustering method
    k_based_methods = {kmeans.k_means, fuzzy_c_means.fuzzy_c_means, my_kmedoids.pam_clustering}
    density_based_methods = {DBSCAN.dbscan_clustering, mean_shift.mean_shift_clustering, OPTICS.optics_clustering, HDBSCAN.hdbscan_clustering, DENCLUE.denclue_clustering}
    
    # Extract arguments correctly
    if clustering_func in k_based_methods:
        if len(args) < 2:
            raise ValueError(f"{clustering_func.__name__} requires at least two arguments: (data, k)")
        data, k = args[:2]
        
        # Ensure k is an integer
        if isinstance(k, (list, np.ndarray)) and len(k) == 1:
            k = int(k[0])
        elif not isinstance(k, int):
            raise TypeError(f"Expected k to be an integer, but got {type(k).__name__}: {k}")
    
    elif clustering_func in density_based_methods:
        if len(args) < 1:
            raise ValueError(f"{clustering_func.__name__} requires at least one argument: (data)")
        data = args[0]
    
    #print(f"Data shape before clustering: {data.shape}")
    
    # Handle specific clustering function cases
    if clustering_func == my_kmedoids.pam_clustering:
        build_function = build_function or my_kmedoids.pam_build
        swap_function = swap_function or my_kmedoids.pam_swap
        print(build_function, swap_function)
        result = clustering_func(data, k, build_function, swap_function)
    
    elif clustering_func == kmeans.k_means:
        result = clustering_func(data, k, **kwargs)
    
    elif clustering_func == fuzzy_c_means.fuzzy_c_means:
        result = clustering_func(data, k, **kwargs)
    
    elif clustering_func == DBSCAN.dbscan_clustering:
        result = clustering_func(data, **kwargs)  # No k required
    elif clustering_func == mean_shift.mean_shift_clustering:
        result = clustering_func(data, **kwargs)  # No k required
    elif clustering_func == OPTICS.optics_clustering: 
        result = clustering_func(data, **kwargs) # No k required
    elif clustering_func == HDBSCAN.hdbscan_clustering: 
        result = clustering_func(data, **kwargs) # No k required
    elif clustering_func == DENCLUE.denclue_clustering: 
        result = clustering_func(data, **kwargs) # No k required
    
    else:
        raise ValueError(f"Unsupported clustering function: {clustering_func.__name__}")
    
    runtime = time.time() - start_time
    print(f"Runtime for {clustering_func.__name__}: {runtime:.6f} seconds")
    
    return result, runtime