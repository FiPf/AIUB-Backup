import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
import calculations

def sort_for_sources(data: np.array, sources: np.array):
    """sorts the data according to the source of the space debris parts

    Args:
        sources (np.array): list of sources of the data

    Returns:
        TLE (np.array): list of all TLE objects
        fragments (np.array): list of all fragment objects
        rest (np.array): all other objects that are not TLE and not fragments
    """
    TLE = []
    fragments = []
    rest = []
    
    for i in range(len(sources)): 
        if sources[i] == 4: 
            TLE.append(data[i])  # Append the i-th element from data
        elif sources[i] == 1:
            fragments.append(data[i])  # Append the i-th element from data
        else: 
            rest.append(data[i])
    
    return np.array(TLE), np.array(fragments), np.array(rest)

def sort_for_sources_all_data(all_data: np.array, source_index: int): 
    """sorts the entire data (objects with all attributes) according to the source of the space debris parts

    Args:
        all_data (np.array): data (for example from array extender)
        source_index (int): index where in data the sources are stored

    Returns:
        TLE (np.array): list of all TLE objects
        fragments (np.array): list of all fragment objects
        rest (np.array): all other objects that are not TLE and not fragments
    """
    TLE = []
    fragments = []
    rest = []
    
    all_data = np.array(all_data)
    
    sources = all_data[source_index]
    
    for i in range(len(sources)): 
        if sources[i] == 4: 
            TLE.append(all_data[:, i])  
        elif sources[i] == 1:
            fragments.append(all_data[:, i]) 
        else: 
            rest.append(all_data[:, i])
    
    return np.array(TLE), np.array(fragments), np.array(rest)


def sort_for_apogee(sem_major: np.array, ecc: np.array, *arrays): 
    """remove objects from data with apogee > 10000 km. Reason: when objects in earth shadow, magnitude is 0

    Args:
        sem_major (np.array): semi major axis of objects
        ecc (np.array): eccentricity of objects
        arrays: multiple arrays to be sorted

    Raises:
        ValueError: when the lengths of the inputs arrays do not align, likely given the wrong input data
        ValueError: when the length of the inputs arrays do not align, likely given the wrong input data

    Returns:
        sorted_arrays (np.arrays): arrays without the objects with apogee > 10'000 km
    """
    if len(sem_major) != len(ecc):
        raise ValueError("The lengths of sem_major and ecc must be the same.") #overkill, should not happen
    
    apogee = [a * (1 + e) for a, e in zip(sem_major, ecc)] #create array with apogees, calculate apogee from e and a

    sorted_arrays = []
    for array in arrays:
        if len(array) != len(apogee):
            raise ValueError("All input arrays must have the same length as sem_major and ecc.")
        sorted_array = [array[i] for i in range(len(apogee)) if apogee[i] > 10000]
        sorted_arrays.append(sorted_array)
    return sorted_arrays

def sort_for_apogee_all_data(all_data: np.array, semi_major_index: int, ecc_index: int):
    """
    Sorts the entire data (objects with all attributes) and removes objects with an apogee greater than 10,000 km.

    Args:
        all_data (np.array): data (for example from array extender)
        semi_major_index (int): index where in data the semi-major axes are stored
        ecc_index (int): index where in data the eccentricities are stored

    Returns:
        filtered_data (np.array): data array without the objects with apogee > 10,000 km
    """
    all_data = np.array(all_data)
    
    # Extract semi-major axis and eccentricity arrays
    semi_major = all_data[semi_major_index, :].astype(float)
    ecc = all_data[ecc_index, :].astype(float)
    
    if len(semi_major) != len(ecc):
        raise ValueError("The lengths of semi_major and ecc must be the same.")

    # Calculate apogee from semi-major axis and eccentricity
    apogee = semi_major * (1 + ecc)

    # Create a list to store indices of objects with apogee <= 10,000 km
    valid_indices = [i for i in range(len(apogee)) if apogee[i] >= 10000]

    # Filter all_data to only include objects with valid apogees
    filtered_data = all_data[:, valid_indices]

    return filtered_data


def sort_for_inclination(inc: np.array, max_inc: float, *arrays):
    """removes all objects from the data that have an inclination higher than a given i

    Args:
        inc (np.array): inclination value array
        max_inc (float): desired maximum value for i 
        arrays (np.arrays): multiple arrays to be sorted

    Raises:
        ValueError: when the lengths of the inputs arrays do not align, likely given the wrong input data

    Returns:
        sorted_arrays (np.arrays): arrays without the objects with inclination > max_inc
    """
    inc = np.where((np.array(inc) >= -1) & (np.array(inc) < 0), 0, inc)
    sorted_arrays = []
    for array in arrays: 
        if len(array) != len(inc): 
            raise ValueError("All input arrays must have the same length as inc!")
        sorted_array = [array[i] for i in range(len(inc)) if inc[i] < max_inc]
        sorted_arrays.append(sorted_array)
    return sorted_arrays

def sort_for_inclination_all_data(all_data: np.array, inc_index: int, max_inc: float):
    """
    Sorts the entire data (objects with all attributes) and removes objects with an inclination greater than max_inc.

    Args:
        all_data (np.array): data (for example from array extender)
        inc_index (int): index where in data the inclinations are stored
        max_inc (float): desired maximum value for inclination

    Returns:
        filtered_data (np.array): data array without the objects with inclination > max_inc
    """
    # Extract inclination array
    inc = all_data[inc_index]
    inc = all_data[inc_index].astype(float)
    # Handle invalid inclination values by setting them to 0
    inc = np.where((np.array(inc) >= -1) & (np.array(inc) < 0), 0, inc)
    # Create a list to store indices of objects with inclination <= max_inc
    valid_indices = [i for i in range(len(inc)) if inc[i] < max_inc]

    # Filter all_data to only include objects with valid inclinations
    filtered_data = all_data[:, valid_indices]

    return filtered_data

def sort_for_magnitudes(mag: np.array, min_mag: float, *arrays, max_mag: float = None): 
    """remove all objects outside a given magnitude range [min_mag, mag_mag]

    Args:
        mag (np.array): array of magnitudes to sort
        min_mag (float): Minimum magnitude, objects with lower magnitudes are removed
        *arrays (np.array): any number of numpy arrays, from which the objects with the indices that were removed in mag are removed
        max_mag (float, optional): Maximum magnitude, objects with higher magnitudes are removed. Defaults to None.

    Raises:
        ValueError: input array must have the same length as the magnitudes array

    Returns: numpy array of input arrays with the right objects removed
        
    """
    sorted_arrays = []
    for array in arrays: 
        if len(array) != len(mag): 
            raise ValueError("All input arrays must have the same length as mag!")
        
        if max_mag is None:
            # Only min_mag filtering
            sorted_array = [array[i] for i in range(len(mag)) if mag[i] >= min_mag]
        else:
            # Filter between min_mag and max_mag inclusive
            sorted_array = [array[i] for i in range(len(mag)) if min_mag <= mag[i] <= max_mag]
        
        sorted_arrays.append(sorted_array)
    return sorted_arrays

def sort_for_sizes(size: np.array, min_size: float, *arrays, max_size: float = None):
    """remove all objects outside a given size range [min_size, mag_size]

    Args:
        mag (np.array): array of sizes to sort
        min_mag (float): Minimum size, smaller objects are removed
        *arrays (np.array): any number of numpy arrays, from which the objects with the indices that were removed in mag are removed
        max_mag (float, optional): Maximum size, larger objects are removed. Defaults to None.

    Raises:
        ValueError: input array must have the same length as the size array

    Returns: numpy array of input arrays with the right objects removed
        
    """
    sorted_arrays = []
    for array in arrays:
        if len(array) != len(size):
            raise ValueError("All input arrays must have the same length as size!")
        
        if max_size is None:
            # Only min_size filtering
            sorted_array = [array[i] for i in range(len(size)) if size[i] >= min_size]
        else:
            # Filter between min_size and max_size inclusive
            sorted_array = [array[i] for i in range(len(size)) if min_size <= size[i] <= max_size]
        
        sorted_arrays.append(sorted_array)
    return sorted_arrays

def sort_for_magnitudes_all_data(all_data: np.array, mag_index: int, min_mag: float):
    """
    Sorts the entire data (objects with all attributes) and removes objects with magnitudes lower than min_mag.

    Args:
        all_data (np.array): data (for example from array extender)
        mag_index (int): index where in data the magnitudes are stored
        min_mag (float): desired minimum magnitude

    Returns:
        filtered_data (np.array): data array without the objects with magnitude < min_mag
    """
    # Extract magnitude array
    mag = all_data[mag_index]
    mag = all_data[mag_index].astype(float)

    # Create a list to store indices of objects with magnitude >= min_mag
    valid_indices = [i for i in range(len(mag)) if mag[i] <= min_mag]

    # Filter all_data to only include objects with valid magnitudes
    filtered_data = all_data[:, valid_indices]

    return filtered_data

def find_real_TLE(TLE_det_GEO: np.array, TLE_det_GTO: np.array, TLE_det_fol: np.array, semi_major_index: int): 
    """Takes all TLE data sorted in surveys, looks for GTO objects according to orbit parameters and sorts the 
    found GTO objects into the survey in which they were found.

    Args:
        TLE_det_GEO (np.array): TLE objects from GEO survey
        TLE_det_GTO (np.array): TLE objects from GTO survey
        TLE_det_fol (np.array): TLE objects from follow-up survey
        semi_major_index (int): Index of semi-major axis in the TLE data array

    Returns:
        add_to_GEO (np.array): TLE objects that are GTO but were found in GEO survey
        add_to_GTO (np.array): TLE objects that are GTO and were found in GTO survey
        add_to_fol (np.array): TLE objects that are GTO but were found in follow-up survey
    """
    TLE_arrays = [TLE_det_GEO, TLE_det_GTO, TLE_det_fol]  # Detected TLE objects for different orbit types
    real_TLE_in_GTO = []
    index_list = [0]

    for array in TLE_arrays: 
        real_TLE_this_array = []
        semi_major = array[semi_major_index]  # Get semi-major axes
        for i in range(len(semi_major)): 
            if semi_major[i] > 43000 or semi_major[i] < 40000:  # Criterion for GTO
                real_TLE_this_array.append(array[:, i])# If criterion is fulfilled, add objects
        real_TLE_in_GTO.extend(real_TLE_this_array)
        index_list.append(index_list[-1] + len(real_TLE_this_array))
    
    # Convert list to numpy array for easier slicing
    real_TLE_in_GTO = np.array(real_TLE_in_GTO).T  # Transpose to align with original data structure
    
    # Slice based on index list
    add_to_GEO = real_TLE_in_GTO[:, :index_list[1]] if index_list[1] > 0 else np.array([])
    add_to_GTO = real_TLE_in_GTO[:, index_list[1]:index_list[2]] if index_list[2] > index_list[1] else np.array([])
    add_to_fol = real_TLE_in_GTO[:, index_list[2]:] if index_list[3] > index_list[2] else np.array([])

    return add_to_GEO, add_to_GTO, add_to_fol

def find_real_TLE_from_corr_obs(data: np.array, semi_major_index: int):
    """Takes all TLE data and decides if the objects are from GTO according to orbital parameters

    Args:
        data (np.array): TLE input data
        semi_major_index (int): index where in the data the semi major axis is stored

    Returns:
        real_TLE (np.array): GTO objects in TLE
    """
    data = np.array(data)
    real_TLE = []
    semi_major = data[semi_major_index]
    semi_major = [float(a) for a in semi_major]
    for i in range(len(semi_major)):
        if semi_major[i] > 45000 or semi_major[i] < 25000:  # Criterion for GTO
                real_TLE.append(data[:, i])
                
    real_TLE = np.array(real_TLE).T
    
    return real_TLE
    
def data_sorter(array: np.array, semi_major_index: int, ecc_index: int, inc_index: int, mag_index: int, source_index: int = None, size_index: int = None):
    """
    Sorts the input data array for apogee, inclination, and magnitudes.
    Optionally sorts by source and splits the data into TLE, fragments, and rest.

    Parameters:
        array (np.array): The input data array, shape (features, samples).
        semi_major_index (int): Index of the semi-major axis in the array.
        ecc_index (int): Index of the eccentricity in the array.
        inc_index (int): Index of the inclination in the array.
        mag_index (int): Index of the magnitudes in the array.
        source_index (int, optional): Index of the source in the array.

    Returns:
        tuple: Sorted arrays (TLE, fragments, rest) if `source_index` is provided.
               Otherwise, returns the sorted array.
    """
    # Ensure float conversion for sorting
    array[semi_major_index] = [float(i) for i in array[semi_major_index]]
    array[ecc_index] = [float(i) for i in array[ecc_index]]
    array[mag_index] = [float(i) for i in array[mag_index]]

    # Sort by apogee
    semi_major = array[semi_major_index]
    ecc = array[ecc_index]
    apogee = semi_major * (1 + ecc)
    sorted_indices_apogee = np.argsort(apogee)
    array = array[:, sorted_indices_apogee]

    # Sort by inclination
    inclinations = array[inc_index]
    max_inc = 40
    valid_indices_inc = [i for i, inc in enumerate(inclinations) if inc < max_inc]
    array = array[:, valid_indices_inc]

    # Sort by magnitudes
    magnitudes = array[mag_index]
    min_mag = 14
    max_mag = 19
    valid_indices_mag = [i for i, mag in enumerate(magnitudes) if min_mag < mag < max_mag]
    array = array[:, valid_indices_mag]

    if size_index is not None:
        min_size = 0.1 #10 cm or 0,1 m
        sizes = array[size_index].astype(float)
        valid_indices_size = [i for i, size in enumerate(sizes) if size >= min_size]
        array = array[:, valid_indices_size]
    # Sort by source if source_index is provided
    if source_index is not None:
        array[source_index] = [int(i) for i in array[source_index]]
        TLE_indices = np.where(array[source_index] == 0)[0]
        frag_indices = np.where(array[source_index] == 1)[0]
        rest_indices = np.where(array[source_index] == 2)[0]

        array_TLE = array[:, TLE_indices]
        array_frag = array[:, frag_indices]
        array_rest = array[:, rest_indices]

        return array_TLE, array_frag, array_rest

    # If no source sorting, return the final array
    return array

def separate_data_by_custom_intervals(input_file, output_dir, intervals):
    """
    Separate the data from an input file into different files based on custom year intervals.
    
    Parameters:
    input_file (str): Path to the input file containing data.
    output_dir (str): Directory where the separated files will be saved.
    intervals (list of int): List of year intervals for splitting the data (e.g., [3, 5, 2]).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the data from the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    interval_data = {}
    start_year = None
    
    for line in lines:
        # Split the line into columns (assuming columns are space-separated)
        columns = line.split()
        
        # Extract MJD and convert to date
        try:
            mjd = float(columns[2])
            date_str = calculations.mjd_to_date(mjd)
            year = int(date_str.split('-')[0])  # Extract year from date string
        except ValueError:
            continue  # Skip lines that don't contain valid MJD
        
        # Determine the start year if not set
        if start_year is None:
            start_year = year
        
        current_interval_start = start_year
        current_interval_end = start_year
        
        # Find the appropriate interval for the current year
        for interval in intervals:
            current_interval_end = current_interval_start + interval - 1
            if current_interval_start <= year <= current_interval_end:
                interval_key = f"{current_interval_start}-{current_interval_end}"
                break
            current_interval_start = current_interval_end + 1
        
        # Append the line to the corresponding interval's data
        if interval_key not in interval_data:
            interval_data[interval_key] = []
        interval_data[interval_key].append(line.strip())
    
    # Write each interval's data to a separate file
    for interval_key, lines in interval_data.items():
        output_file = os.path.join(output_dir, f'6param_{interval_key}.txt')
        with open(output_file, 'w') as file:
            for line in lines:
                file.write(f'{line}\n')

def remove_zero_background_mag(data: np.array, background_mag_index: int, mag_index: int, illumination_index: int) -> np.array:
    """
    Removes objects with a background magnitude (mag_backgr) equal to 0.000 and objects with an object magnitude of 99 or more.
    Function was used to compare crossings to plugin output.

    Args:
        data (np.array): Crossing or detection data.
        background_mag_index (int): Index of the background magnitude in the data array.
        mag_index (int): Index of the object magnitude in the data array.

    Returns:
        np.array: Filtered numpy array with objects removed where background magnitude is 0.000 or object magnitude is 90 or more.
    """
    
    # Filter out objects with background magnitude equal to 0.000
    mask_background = (data[background_mag_index] != 0.000) #& (data[background_mag_index] < 21.3)
    filtered_data = data[:, mask_background]  # Apply mask to all rows
    
    # Filter out objects with object magnitude 90 or more
    #mask_magnitude = filtered_data[mag_index] < 50
    #filtered_data = filtered_data[:, mask_magnitude]  # Apply second mask

    #Filter out objects with objects with illumination zero
    #mask_illumination = filtered_data[illumination_index] != 0
    #filtered_data = filtered_data[:, mask_illumination]
    
    mag_obj_before = data[mag_index]
    mag_backgr_before = data[background_mag_index]
    
    mag_obj_after = filtered_data[mag_index]
    mag_backgr_after = filtered_data[background_mag_index]
    
    #plot used for debugging, Kreisbahnproblem
    """fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(mag_backgr_before, mag_obj_before, s=5, alpha=0.5, label="Before Filtering")
    ax[0].set_xlabel("Background Magnitude")
    ax[0].set_ylabel("Object Magnitude")
    ax[0].set_title("Before Filtering")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].scatter(mag_backgr_after, mag_obj_after, s=5, alpha=0.5, color="red", label="After Filtering")
    ax[1].set_xlabel("Background Magnitude")
    ax[1].set_ylabel("Object Magnitude")
    ax[1].set_title("After Filtering")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()"""
    
    return filtered_data

def inclination_filter(crs_data, celmech_data, celmech_fail_mask, max_inclination):
    """
    Filter out objects with inclinations larger than maxinc from the crossing dataset,
    and remove the corresponding objects from the Celmech dataset.
    
    The filtering is done in order. For each object (i.e. each column in crs_data),
    if the corresponding celmech_fail_mask value indicates a failure (== 1), then that
    object is skipped (i.e. not included in the filtered CRS dataset). For objects with 
    successful orbit determinations (celmech_fail_mask == 0), if their crossing inclination 
    (at row incl_index) is greater than maxinc, they are filtered out in both datasets.
    
    Args:
        crs_data (np.ndarray): Crossing dataset with shape (num_params, num_objects).
        celmech_data (np.ndarray): Celmech dataset with shape (num_objects, 4).
        celmech_fail_mask (np.ndarray): 1D array of length num_objects (0 for success, 1 for failure).
        maxinc (float): Maximum allowed inclination.
        incl_index (int, optional): Index of the inclination in the CRS dataset. Defaults to 9.
    
    Returns:
        filtered_crs (np.ndarray): Filtered crossing data (same shape convention: parameters x objects).
        filtered_celmech (np.ndarray): Filtered Celmech data (rows corresponding to the kept objects).
    """
    # Ensure all arrays have the same length
    min_length = min(len(crs_data[9]), len(celmech_data), len(celmech_fail_mask))
    
    # Trim datasets if needed
    crs_data = [arr[:min_length] for arr in crs_data]  # Trim each CRS data array
    celmech_data = celmech_data[:min_length]
    celmech_fail_mask = celmech_fail_mask[:min_length]
    
    # Apply the inclination filter
    crs_inclination = crs_data[9]  # Assuming inclination is at index 9
    mask = (crs_inclination <= max_inclination) | (celmech_fail_mask == 1)
    # This keeps objects with inclination <= max_inclination OR failed Celmech objects

    # Apply mask to both CRS and Celmech data
    filtered_crs = [arr[mask] for arr in crs_data]
    filtered_celmech = celmech_data[mask]

    # Return filtered datasets
    return filtered_crs, filtered_celmech[:, 2], filtered_celmech[:, 3]  # Return I and Node from Celmech
