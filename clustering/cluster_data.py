from collections import namedtuple
import numpy as np
import time 
import my_kmedoids
import kmeans
import fuzzy_c_means
from clustering_utils import ClusteringResult, ClusterData
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

#change the ClusterData namedtuple if you want to add more dimensions
#ClusterData = namedtuple("ClusterData", ["inc", "raan"])
#ClusterData = namedtuple("ClusterData", ["inc", "raan", "ecc"]) #later we can add the eccentricity

def run_clustering(algorithm, name, data, data_min, data_max, *args, **kwargs):
    print(f"args received in run_clustering: {args}, type: {type(args)}")

    metrics = []  # List containing all the scores, 2D standard deviation, cluster densities
    plot = kwargs.pop("plot", True)  # Default to True if not provided

    print(f"\n{name} result:")
    
    # Initialize progress bar
    with tqdm(total=3, desc=f"Running {name}", unit="step") as pbar:
        result, runtime = estimate_runtime(algorithm, data, *args, **kwargs)
        pbar.update(1)  # Step 1: Clustering done

        # Calculate the number of clusters and the number of points per cluster
        n_clusters = len(set(result.labels))  # Number of unique labels (clusters)
        points_per_cluster = {i: list(result.labels).count(i) for i in set(result.labels)}  # Count points per cluster

        if n_clusters > 1:
            # Calculate all the metrics
            DB_sc = scores.DB_score(result)
            CH_sc = scores.CH_score(result)
            dunn_index_sc = scores.dunn_index_score(result)
            sil_sc = scores.sil_score(result)
            cluster_std = scores.cluster_std_eigen(result)
            square_density = scores.cluster_density_squares(result)
            hull_density = scores.cluster_density_convex_hull(result)
            metrics = [DB_sc, CH_sc, dunn_index_sc, sil_sc, cluster_std, square_density, hull_density]
        
        pbar.update(1)  # Step 2: Metrics computed

        # Unnormalize cluster centers
        unnormalized_data, cluster_centers = unnormalize(result.data, result.cluster_centers, data_min, data_max)
        if plot: 
            plotter = ClusterPlotter(unnormalized_data, result.labels, cluster_centers)
            plotter.clusters_2d_plot(f"{name} - 2D Cluster Visualization")
        
        pbar.update(1)  # Step 3: Plotting done (if enabled)

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
    unnormalized_centers = None
    unnormalized_data = normalized_data * (data_max - data_min) + data_min
    if cluster_centers is not None:
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
    return ClusterData(inc=inc, raan=raan)#, ecc=ecc) #later

def bin_finder(num_years_per_bin: int, overlap_years: int):
    """Create bins of years with a specified overlap. used as helper function in bin_data.

    Args:
        num_years_per_bin (int): Number of years in each bin.
        overlap_years (int): Number of overlapping years between bins.

    Returns:
        list of np.ndarray: year ranges according to the binning. 
    """
    start_year = 2002
    end_year = 2023

    year_ranges = []
    current_start = start_year

    while current_start < end_year:
        bin_end = min(current_start + num_years_per_bin, end_year)
        year_ranges.append(np.arange(current_start, bin_end))
        current_start += num_years_per_bin - overlap_years  # Move start forward considering overlap

        if bin_end == end_year:
            break

    return year_ranges

def find_filename(year: str, seed: str, orbit: str, crs_det: str, population_type: PopulationType):
    """find the right filename for the data. used as helper function in bin_data

    Args:
        year (str): year of the data
        seed (str): seed of the data
        orbit (str): orbit type, geo, gto or followup
        crs_det (str): if crs, then the crossing data is extracted. if det, then the detections data is extracted. 
        population_type (PopulationType): which batch of simulation output to use. 

    Returns:
        file (str): file with complete path to the data
    """
    year2 = year[2:]
    
    if int(year2) < (18 if population_type == PopulationType.NEWPOP_TH3 else 19):
        suffix = ""
    else:
        #suffix = f"_{population_type.value}"
        suffix = population_type.value if population_type.value else ""

    file = f"stat_Master_{year2}_{orbit}_s{seed}{suffix}.{crs_det}"
    file = os.path.join("..","input", file)

    return file

def bin_data(num_years_per_bin: int, overlap_years: int, crs_det: str, population_type: PopulationType):
    """bin the data according to specific bin sizes and overlaps. 

    Args:
        num_years_per_bin (int): number of years per bin
        overlap_years (int): overlapping years between the bins
        crs_det (str): if crs, then the crossing data is extracted. if det, then the detections data is extracted. 
        population_type (PopulationType): which batch of simulation output to use.

    Returns:
        data_batches (dict): contains the data ordered in the bins, ready for processing. 
    """    
    year_ranges = bin_finder(num_years_per_bin, overlap_years)
    orbit_types = ["geo", "gto", "fol", "all"]  
    data_batches = {}
    seed = "1"

    for years in year_ranges: 
        bins = []
        combined_inc = []
        combined_raan = []
        
        for year in years:
            sub_data = []
            
            for orbit in ["geo", "gto", "fol"]:
                filename = find_filename(str(year), seed, orbit, crs_det, population_type)
                data = prepare_data_for_clustering(filename)
                sub_data.append(data)

            if "all" in orbit_types: 
                combined_inc.extend(np.concatenate([d.inc for d in sub_data]))
                combined_raan.extend(np.concatenate([d.raan for d in sub_data]))
                bins.append(ClusterData(inc=np.array(combined_inc), raan=np.array(combined_raan)))
            else:
                bins.extend(sub_data)

        data_batches[tuple(years)] = bins

    return data_batches

def estimate_runtime(clustering_func, *args, build_function=None, swap_function=None, **kwargs):
    """
    Measures execution time of a clustering function, handling different clustering algorithms.
    """
    start_time = time.time()
    
    # Identify the type of clustering method
    k_based_methods = {kmeans.k_means, fuzzy_c_means.fuzzy_c_means, my_kmedoids.pam_clustering}
    density_based_methods = {DBSCAN.dbscan_clustering, mean_shift.mean_shift_clustering}
    
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
    
    print(f"Data shape before clustering: {data.shape}")
    
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
    
    else:
        raise ValueError(f"Unsupported clustering function: {clustering_func.__name__}")
    
    runtime = time.time() - start_time
    print(f"Runtime for {clustering_func.__name__}: {runtime:.6f} seconds")
    
    return result, runtime