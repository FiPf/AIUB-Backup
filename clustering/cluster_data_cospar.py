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

# --- Updated Namedtuples including cospar_id ---
ClusteringResult = namedtuple("ClusteringResult", ["labels", "cluster_centers", "data", "cospar_id"])

ClusterData = namedtuple("ClusterData", [
    "cospar_id",  # metadata for coloring only, excluded from clustering
    "ecc", "sem_maj", "inc", "raan", "perigee", "true_lat",
    "mean_motion", "mag_obj", "diameter"
])

# Features used for clustering (exclude cospar_id)
CLUSTER_FEATURES = [
    "ecc", "sem_maj", "inc", "raan", "perigee",
    "true_lat", "mean_motion", "mag_obj", "diameter"
]

def cluster_data_to_array(data: ClusterData) -> np.ndarray:
    """Convert ClusterData to numpy array of features only (exclude cospar_id).

    Args:
        data (ClusterData): input data

    Returns:
        np.ndarray: array of shape (n_samples, n_features)
    """
    return np.vstack([getattr(data, f) for f in CLUSTER_FEATURES]).T

def normalize_data(arr: np.array):
    """Normalize data to [0,1] using MinMaxScaler.

    Args:
        arr (np.array): Input data array.

    Returns:
        normalized_data (np.array): Normalized data.
        data_min (np.array): Minimum values before normalization.
        data_max (np.array): Maximum values before normalization.
    """    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(arr)
    return normalized_data, scaler.data_min_, scaler.data_max_

def unnormalize(normalized_data: np.array, cluster_centers: np.array, data_min: np.array, data_max: np.array, labels=None): 
    """Revert normalization to original scale.

    Args:
        normalized_data (np.array): Normalized data.
        cluster_centers (np.array): Normalized cluster centers.
        data_min (np.array): Min values used in normalization.
        data_max (np.array): Max values used in normalization.
        labels (np.array, optional): Cluster labels.

    Returns:
        unnormalized_data (np.array): Data back to original scale.
        unnormalized_centers (np.array or None): Cluster centers in original scale or None if not given.
    """
    unnormalized_data = normalized_data * (data_max - data_min) + data_min
    unnormalized_centers = None
    if cluster_centers is not None and cluster_centers.size > 0:
        unnormalized_centers = cluster_centers * (data_max - data_min) + data_min
    elif labels is not None:
        # Compute centers as mean of points in each cluster
        unique_labels = np.unique(labels)
        centers = []
        for label in unique_labels:
            centers.append(np.mean(unnormalized_data[labels == label], axis=0))
        unnormalized_centers = np.vstack(centers) if centers else None
    return unnormalized_data, unnormalized_centers

def adjust_raan_range(raan_values):
    """Convert RAAN from [0, 360]° to [-180, 180]°.

    Args:
        raan_values (np.array or list): RAAN values in degrees

    Returns:
        np.array: RAAN values adjusted to [-180, 180]
    """    
    raan_adjusted = np.mod(np.array(raan_values) + 180, 360) - 180
    return raan_adjusted

def prepare_data_for_clustering(filenames: list) -> ClusterData:
    """Load data from file, perform sorting, adjust RAAN, compute mean motion, prepare ClusterData.

    Args:
        filename (list): paths to .crs or .det files

    Returns:
        ClusterData: ready for clustering, includes cospar_id and features
    """
    data = []

    for filename in filenames:
        data_one_file = getdata.array_extender(filename)
        data_one_file = np.array(data_one_file)
        data_TLE, data_frag, data_rest = sortdata.data_sorter(data_one_file, semi_major_index=8, ecc_index=10, inc_index=9, mag_index=20, source_index=3)
        data_one_file = np.hstack([data_frag, data_rest])
        data.append(data_one_file)

    data = np.hstack(data)
    cospar_id = data[0]

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
    mean_motion = np.sqrt(mu/(sem_maj*1000)**3) # convert km to m
    mean_motion = mean_motion/(2*np.pi)*86400

    arrays = [inc, raan, ecc, perigee, mag_obj, sem_maj, diameter, true_lat, mean_motion]
    filtered, no_match = [[] for _ in arrays], [[] for _ in arrays]
    final_cospars = []

    cospar_dict = getdata.read_metadata_file("../input/geogto.dat")
    print(cospar_dict.keys())

    all_ids = []
    for i, obj in enumerate(cospar_id): 
        prefix = str(int(obj))[:3]
        id = cospar_dict.get(int(prefix))
        all_ids.append(id)
        if id: 
            for j, arr in enumerate(arrays): 
                filtered[j].append(arr[i])
            final_cospars.append(id)

        else: 
            for j, arr in enumerate(arrays): 
                no_match[j].append(arr[i])

    filtered = [np.array(a) for a in filtered]
    no_match = [np.array(a) for a in no_match]

    cospar_ids_clean = [cid for cid in all_ids if cid is not None]
    cospar_str = ", ".join(cospar_ids_clean)
    print("All COSPAR-IDS form my geogto.dat file:\n", cospar_str)

    return (
    ClusterData(  
        cospar_id=np.array(final_cospars),
        inc=filtered[0],
        raan=filtered[1],
        ecc=filtered[2],
        perigee=filtered[3],
        mag_obj=filtered[4],
        sem_maj=filtered[5],
        diameter=filtered[6],
        true_lat=filtered[7],
        mean_motion=filtered[8]
    ),
    ClusterData( 
        cospar_id=np.array(['Other'] * len(no_match[0])),
        inc=no_match[0],
        raan=no_match[1],
        ecc=no_match[2],
        perigee=no_match[3],
        mag_obj=no_match[4],
        sem_maj=no_match[5],
        diameter=no_match[6],
        true_lat=no_match[7],
        mean_motion=no_match[8]
    )
)


def merge_cluster_data(cluster_data_list: list):
    """Merge multiple ClusterData objects into one.

    Args:
        cluster_data_list (list): List of ClusterData objects

    Returns:
        ClusterData: merged data containing concatenated arrays
    """
    cospar_id_all = np.concatenate([cd.cospar_id for cd in cluster_data_list])
    sem_maj_all = np.concatenate([cd.sem_maj for cd in cluster_data_list])
    inc_all = np.concatenate([cd.inc for cd in cluster_data_list])
    ecc_all = np.concatenate([cd.ecc for cd in cluster_data_list])
    raan_all = np.concatenate([cd.raan for cd in cluster_data_list])
    perigee_all = np.concatenate([cd.perigee for cd in cluster_data_list])
    true_lat_all = np.concatenate([cd.true_lat for cd in cluster_data_list])
    mean_motion_all = np.concatenate([cd.mean_motion for cd in cluster_data_list])
    mag_obj_all = np.concatenate([cd.mag_obj for cd in cluster_data_list])
    diameter_all = np.concatenate([cd.diameter for cd in cluster_data_list])

    return ClusterData(
        cospar_id=cospar_id_all,
        ecc=ecc_all,
        sem_maj=sem_maj_all,
        inc=inc_all,
        raan=raan_all,
        perigee=perigee_all,
        true_lat=true_lat_all,
        mean_motion=mean_motion_all,
        mag_obj=mag_obj_all,
        diameter=diameter_all
    )

def bin_data_for_clustering(year_ranges: dict, print_res: bool = True):
    """Load, bin and merge data by year ranges for clustering.

    Args:
        year_ranges (dict): dict of str->np.array(years)
        print_res (bool): print file info for debug

    Returns:
        list of (ClusterData, year_range) tuples
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
        cospar_all = []

        for orbit, files in orbits.items():
            cluster_data_pairs = [prepare_data_for_clustering(f) for f in files]
            cluster_data_list = [cd for pair in cluster_data_pairs for cd in pair]
            merged_data = merge_cluster_data(cluster_data_list)

            cospar_all.append(merged_data.cospar_id)
            diameter_all.append(merged_data.diameter)
            sem_maj_all.append(merged_data.sem_maj)
            inc_all.append(merged_data.inc)
            ecc_all.append(merged_data.ecc)
            raan_all.append(merged_data.raan)
            mag_obj_all.append(merged_data.mag_obj)
            perigee_all.append(merged_data.perigee)
            true_lat_all.append(merged_data.true_lat)
            mean_motion_all.append(merged_data.mean_motion)

        cospar_all = np.concatenate(cospar_all) if cospar_all else np.array([])
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
            ClusterData(
                cospar_id=cospar_all,
                ecc=ecc_all,
                sem_maj=sem_maj_all,
                inc=inc_all,
                raan=raan_all,
                perigee=perigee_all,
                true_lat=true_lat_all,
                mean_motion=mean_motion_all,
                mag_obj=mag_obj_all,
                diameter=diameter_all
            ), 
            year_range
        ))

    return results

def estimate_runtime(clustering_func: Callable, *args, build_function: Callable=None, swap_function: Callable=None, **kwargs):
    """Measures execution time of a clustering function.

    Args:
        clustering_func (Callable): clustering function
        build_function (Callable, optional): PAM/kmedoids build function
        swap_function (Callable, optional): PAM/kmedoids swap function

    Returns:
        (ClusteringResult, float): result and runtime in seconds
    """    
    start_time = time.time()
    
    k_based_methods = {kmeans.k_means, fuzzy_c_means.fuzzy_c_means, my_kmedoids.pam_clustering}
    density_based_methods = {DBSCAN.dbscan_clustering, mean_shift.mean_shift_clustering, OPTICS.optics_clustering, HDBSCAN.hdbscan_clustering, DENCLUE.denclue_clustering}
    
    if clustering_func in k_based_methods:
        if len(args) < 2:
            raise ValueError(f"{clustering_func.__name__} requires at least two arguments: (data, k)")
        data, k = args[:2]
        if isinstance(k, (list, np.ndarray)) and len(k) == 1:
            k = int(k[0])
        elif not isinstance(k, int):
            raise TypeError(f"Expected k to be int, got {type(k).__name__}: {k}")
    
    elif clustering_func in density_based_methods:
        if len(args) < 1:
            raise ValueError(f"{clustering_func.__name__} requires at least one argument: (data)")
        data = args[0]
    
    if clustering_func == my_kmedoids.pam_clustering:
        build_function = build_function or my_kmedoids.pam_build
        swap_function = swap_function or my_kmedoids.pam_swap
        result = clustering_func(data, k, build_function, swap_function)
    
    elif clustering_func == kmeans.k_means:
        result = clustering_func(data, k, **kwargs)
    
    elif clustering_func == fuzzy_c_means.fuzzy_c_means:
        result = clustering_func(data, k, **kwargs)
    
    elif clustering_func == DBSCAN.dbscan_clustering:
        result = clustering_func(data, **kwargs)
    elif clustering_func == mean_shift.mean_shift_clustering:
        result = clustering_func(data, **kwargs)
    elif clustering_func == OPTICS.optics_clustering:
        result = clustering_func(data, **kwargs)
    elif clustering_func == HDBSCAN.hdbscan_clustering:
        result = clustering_func(data, **kwargs)
    elif clustering_func == DENCLUE.denclue_clustering:
        result = clustering_func(data, **kwargs)
    
    else:
        raise ValueError(f"Unsupported clustering function: {clustering_func.__name__}")
    
    runtime = time.time() - start_time
    print(f"Runtime for {clustering_func.__name__}: {runtime:.6f} seconds")
    
    return result, runtime

def run_clustering(algorithm: Callable, name: str, data: np.array, data_min: np.array, data_max: np.array, *args, **kwargs):
    """Run clustering algorithm, calculate metrics, and include cospar_id for coloring.

    Args:
        algorithm (Callable): clustering function
        name (str): algorithm name
        data (np.array): normalized data
        data_min (np.array): data min for unnormalizing
        data_max (np.array): data max for unnormalizing

    Returns:
        ClusteringResult, runtime, number_of_clusters, points_per_cluster, metrics
    """
    metrics = []
    metric = kwargs.pop("metric", False)
    cospar_id = kwargs.pop("cospar_id", None)  # pop cospar_id from kwargs if given

    result, runtime = estimate_runtime(algorithm, data, *args, **kwargs)

    n_clusters = len(set(result.labels))
    points_per_cluster = {i: list(result.labels).count(i) for i in set(result.labels)}

    if n_clusters > 1 and metric:
        DB_sc = scores.DB_score(result)
        CH_sc = scores.CH_score(result)
        dunn_index_sc = scores.dunn_index_score(result)
        sil_sc = scores.sil_score(result)
        cluster_std = scores.cluster_std_eigen(result)
        # Commented out, only works for 2D data
        # square_density, square_bounds = scores.cluster_density_squares(result)
        # hull_density, hull_bounds = scores.cluster_density_convex_hull(result)
        metrics = [DB_sc, CH_sc, dunn_index_sc, sil_sc, cluster_std]  # , square_density, hull_density]

    unnormalized_data, unnormalized_centers = unnormalize(result.data, result.cluster_centers, data_min, data_max)

    # Attach cospar_id to the clustering result for coloring only, not used in clustering itself
    clustering_result = ClusteringResult(
        labels=result.labels,
        cluster_centers=unnormalized_centers,
        data=unnormalized_data,
        cospar_id=cospar_id
    )

    return clustering_result, runtime, n_clusters, points_per_cluster, metrics


def run_clustering_dbcv_score(algorithm: Callable, name: str, data: np.array, data_min: np.array, data_max: np.array, *args, **kwargs):
    """Run clustering algorithm and calculate DBCV score and other metrics.

    Args:
        algorithm (Callable): clustering function
        name (str): algorithm name
        data (np.array): normalized data
        data_min (np.array): data min for unnormalizing
        data_max (np.array): data max for unnormalizing

    Returns:
        ClusteringResult, runtime, number_of_clusters, points_per_cluster, metrics
    """
    metrics = []
    plot = kwargs.pop("plot", True)
    cospar_id = kwargs.pop("cospar_id", None)

    result, runtime = estimate_runtime(algorithm, data, *args, **kwargs)

    n_clusters = len(set(result.labels))
    points_per_cluster = {i: list(result.labels).count(i) for i in set(result.labels)}

    if n_clusters > 1:
        print("Calculating DBCV score...")
        dbcv_score = scores.DBCV_score_rust(result)
        cluster_std = scores.cluster_std_eigen(result)
        square_density, square_bounds = scores.cluster_density_squares(result)
        hull_density, hull_bounds = scores.cluster_density_convex_hull(result)
        metrics = [dbcv_score, cluster_std, square_density, hull_density]

    unnormalized_data, unnormalized_centers = unnormalize(result.data, result.cluster_centers, data_min, data_max)

    clustering_result = ClusteringResult(
        labels=result.labels,
        cluster_centers=unnormalized_centers,
        data=unnormalized_data,
        cospar_id=cospar_id
    )

    if plot:
        plotter = ClusterPlotter(unnormalized_data, clustering_result.labels, clustering_result.cluster_centers)
        plotter.clusters_2d_plot(f"{name} - 2D Cluster Visualization")

    return clustering_result, runtime, n_clusters, points_per_cluster, metrics

