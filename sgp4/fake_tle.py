#this file contains all functions to generate a fake tle, which is used for the input of the sgp4 propagator
#functions to extract the data, calculate some parameters for the TLE and to write the TLE
#it takes the epoch from the plugin.pro file and the orbital elements from the *.crs file
import numpy as np
from astropy.time import Time
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath("..")) 
import getdata
from getdata import PopulationType, data_returner
from collections import namedtuple
import os

def compute_b_drag(diameter: np.array, semi_major: np.array, sources: np.array):
    """compute the b star drag term, which is used in the TLE for the orbit propagation. 
    Use area-to-mass ratio from PROOF/ MASTER model and use drag_coeff = 2.2 (standard default value). 
    The density is taken from https://en.wikipedia.org/wiki/BSTAR. 

    Args:
        diameter (np.array): object diameter
        semi_major (np.array): semi major axis of the object
        sources (np.array): sources of the object

    Returns:
        b_star_drag_values (np.array): calculated bstar drag term, can be used in the TLE
    """
    b_star_drag_vals = []
    b_vals = [] 
    area_to_mass_vals = []
    for d, s in zip(diameter, sources): 
        a_m = compute_am(d, s) # two possible versions, see below
        area_to_mass_vals.append(a_m)

    drag_coeff = 2.2
    density_const = 0.15696615
    R_earth = 6378

    for ele, a in zip(area_to_mass_vals, semi_major): 
        rho = density_const * (R_earth / a)
        b_star_drag = drag_coeff*ele*rho*0.5
        b_star_drag_vals.append(b_star_drag)
        b = drag_coeff*ele
        b_vals.append(b)

    return b_star_drag_vals, b_vals

def compute_am(d: float, source: int):
    """compute the area-to-mass ratiom A/m for an object of size d. The method for this function is
    described in the MASTER final report! 

    Args:
        d (float): diameter of the object
        source (int): source of the object (PROOF number 1- 6)

    Returns:
        area_to_mass (float): A/m in m^2/kg
    """    
    object_type = ""

    if source == 1: 
        object_type == 'spacecraft'
    if source == 4: 
        object_type == 'rocket'
    else: 
        object_type == 'None' #small objects only, all sources excpet 1 (=fragments) and 4 (=TLE)

    log_d = np.log10(d)
    small_particle_limit = 1.7 #default value in case that object_type is None
    r_thresh = 0 #default value in case that object_type is None
    
    # Define bridging function threshold only 
    if object_type != 'None': 
        if object_type == 'rocket':
            r_thresh = 10 ** (log_d + 1.76)
            small_particle_limit = 1.7  # cm
        elif object_type == 'spacecraft':
            r_thresh = 10 ** (log_d + 1.05)
            small_particle_limit = 8  # cm
    
    # Generate random r for decision making, draw from a uniform distribution
    r = np.random.uniform(0, 1)
    
    # whether to use small or large particle distribution
    if d <= small_particle_limit / 100:  # Convert cm to meters
        alpha, mu1, sigma1 = 1, -0.3, 0.2 if log_d <= -3.5 else 0.2 + 0.1333 * (log_d + 3.5)
        chi = np.random.normal(mu1, sigma1)
    elif d >= 0.11:
        # Large particle model (bi-modal normal distribution)
        delta = log_d
        if object_type == 'rocket':
            if delta <= -1.4:
                alpha = 1
            elif -1.4 < delta < 0:
                alpha = 1 - 0.3571 * (delta + 1.4)
            else:
                alpha = 0.5
            
            mu1 = -0.45 if delta <= -0.5 else -0.45 - 0.9 * (delta + 0.5)
            mu2, sigma1, sigma2 = -0.9, 0.55, 0.28 if delta <= -1 else 0.28 - 0.1636 * (delta + 1)
            sigma2 = max(0.1, sigma2)
        else:  # Spacecraft
            if delta <= -1.95:
                alpha = 0
            elif -1.95 < delta < 0.55:
                alpha = 0.3 + 0.4 * (delta + 1.2)
            else:
                alpha = 1
            
            mu1 = -0.6 if delta <= -1.1 else -0.6 - 0.318 * (delta + 1.1)
            mu2 = -1.2 if delta <= -0.7 else -1.2 - 1.333 * (delta + 0.7)
            mu2 = max(-2, mu2)
            sigma1 = 0.1 if delta <= -1.3 else 0.1 + 0.2 * (delta + 1.3)
            sigma1 = min(0.3, sigma1)
            sigma2 = 0.5 if delta <= -0.5 else 0.5 - (delta + 0.5)
            sigma2 = max(0.3, sigma2)
        
        # Sample from bi-modal distribution
        if np.random.uniform(0, 1) < alpha:
            chi = np.random.normal(mu1, sigma1)
        else:
            chi = np.random.normal(mu2, sigma2)
    else:
        # Bridging function
        if r > r_thresh:
            return compute_am(0.11, source)
        else:
            return compute_am(small_particle_limit / 100, source)
    
    area_to_mass = 10**chi
    return area_to_mass

#def compute_am(d: float, source: int) -> float:
    """
    Compute the area-to-mass ratio A/m for an object of size d.
    The method follows the MASTER final report.
    
    Args:
        d (float): Diameter of the object (m)
        source (int): Source category (1 = spacecraft, 4 = rocket, else = fragments/MLI)
    
    Returns:
        float: A/m in m²/kg
    """
    if source == 1:  # Spacecraft
        return 0.1 * d**-0.5
    elif source == 4:  # Rocket bodies
        return 0.05 * d**-0.4
    else:  # Fragments and MLI
        return 5.0 * d**-0.8

def data_from_crs_and_det(crs_filename: str, det_filename: str): 
    """extract data from crossing and detections file and store in namedtuples. 

    Args:
        crs_filename (str): filename of the crossing file
        det_filename (str): filename of the detection file

    Returns:
        crsData (namedtuple): namedtuple containing the crossing data
        detData (namedtuple): namedtuple containing the detection data
    """

    crsData = namedtuple("crsData", ["diameter", "sources", "sem_major", "inc", "ecc", "arg_per", "raan", "true_lat", "background_mag"])
    detData = namedtuple("detData", ["diameter", "sources", "sem_major", "inc", "ecc", "arg_per", "raan", "true_lat", "background_mag"])

    data = getdata.array_extender(crs_filename)
    diameter = data[1]
    sources = data[3]
    sem_major = data[8]
    inc = data[9]
    ecc = data[10]
    arg_per = data[11]
    raan = data[12]
    true_lat = data[13]
    background_mag = data[21]

    crsData = crsData(diameter, sources, sem_major, inc, ecc, arg_per, raan, true_lat, background_mag)

    data = getdata.array_extender(det_filename)
    diameter = data[1]
    sources = data[3]
    sem_major = data[8]
    inc = data[9]
    ecc = data[10]
    arg_per = data[11]
    raan = data[12]
    true_lat = data[13]
    background_mag = data[21]

    detData = detData(diameter, sources, sem_major, inc, ecc, arg_per, raan, true_lat, background_mag)

    return crsData, detData

def data_from_plugin(year: str, orbit_type: str):
    """get data from plugin and store in namedtuple

    Args:
        year (str): year of the data
        orbit_type (str): orbit type of the data

    Returns:
        pluginData (namedtuple): namedtuple containing the important values (epoch!)
    """
    pluginData = namedtuple("pluginData", ["epoch"])
    err = False
    ell = False
    year2 = year[2:]
    filename = f"plugin_{year2}_{orbit_type}.pro"
    filename = os.path.join("..", "input", filename)
    print(filename)
    data = getdata.array_extender_plugin(filename)
    epoch = data[2]
    return pluginData(epoch) 

def data_from_celmech(year:str, dir: str, orbit_type: str, err :bool = False, ell: bool = False): 
    """unpack data from celmech output file

    Args:
        year (str): year of the data
        dir (str): where the file is stored
        orbit_type (str): orbit type of the data
        err (bool, optional): celmech calculations with errors. Setting to True makes no sense at all, don't do it. Defaults to False.
        ell (bool, optional): elliptical orbits. Defaults to False.

    Returns:
        failed_mask (list): when Celmech failed to calculate the orbit, 0, else 1
        dates (list): dates extracted from Celmech output file 
        orbit_type_data (dict): dictionary of data from Celmech calculation
    """
    population_type = PopulationType.NORMAL
    total_num_of_objects = 0
    yy = year[2:]

    orbit_type_data_dict = {"num_obs": [], "rms": [], "time_interval": [], "num_iter": [], 
                "P": [], "A": [], "E": [], "I": [], "Node": [], "Per": [], "IPer": []}

    files = []
    file = getdata.get_celmech_OUT_files(year, orbit_type, err, ell) #find the right Celmech file for given year and orbit_type type
    file = os.path.join("..", "input_celmech", file)
    files.append(file)
    orbit_type_data, number_of_obj, dates, failed_mask = getdata.get_orbele_and_date_from_celmech(files) #extract orbit_typeal elements from Celmech files
    
    total_num_of_objects += number_of_obj
    
    # Append the data to the corresponding orbit_type type in the dictionary
    orbit_type_data_dict["num_obs"].append(orbit_type_data[0])
    orbit_type_data_dict["rms"].append(orbit_type_data[1])
    orbit_type_data_dict["time_interval"].append(orbit_type_data[2])
    orbit_type_data_dict["num_iter"].append(orbit_type_data[3])
    orbit_type_data_dict["P"].append(orbit_type_data[4])
    orbit_type_data_dict["A"].append(orbit_type_data[5])
    orbit_type_data_dict["E"].append(orbit_type_data[6])
    orbit_type_data_dict["I"].append(orbit_type_data[7])
    orbit_type_data_dict["Node"].append(orbit_type_data[8])
    orbit_type_data_dict["Per"].append(orbit_type_data[9])
    orbit_type_data_dict["IPer"].append(orbit_type_data[10])            
    
    maxinc = 22 #cut off all data at greater inclinations
    apogee_threshold = 10000  # in kilometers (10k km)

    # Filtering for inclination and apogee
    inc_data = np.array(orbit_type_data_dict["I"])
    node_data = np.array(orbit_type_data_dict["Node"])
    ecc_data = np.array(orbit_type_data_dict["E"])
    a_data = np.array(orbit_type_data_dict["A"])

    mask = inc_data <= maxinc
    orbit_type_data_dict["I"] = inc_data[mask]
    orbit_type_data_dict["Node"] = node_data[mask]
    orbit_type_data_dict["E"] = ecc_data[mask]
    orbit_type_data_dict["A"] = a_data[mask]

    return failed_mask, dates, orbit_type_data

def combine_data(crsData: namedtuple, detData: namedtuple, failed_mask: np.array, dates: np.array, orbit_type_data: np.array, det: bool = True):
    """read out crossing data and celmech data. remove backgroundmagnitude = 0 in crossing data, apply failed mask
    and bring the epoch together with the 6 orbital elements. Apply the detection filter if needed.

    Args:
        crsData (namedtuple): contains the crossing data
        detData (namedtuple): contains the detections data
        failed_mask (np.array): failed mask from failed orbit determination in Celmech
        dates (np.array): dates extracted from Celmech
        orbit_type_data (np.array): celmech data
        det (bool, optional): If True, only detections are written in the TLE, else also crossings. Defaults to True.

    Returns:
        filtered_crsData (namedtuple): crossing data filtered as desired. If det = True, only detected objects
        filtered_orbit_type_data (dict): celmech data, filtered as desired. 
    """    
    # Convert to NumPy arrays
    background_mask = np.array(crsData.background_mag) != 0  # Remove background_mag = 0
    valid_mask = (np.array(failed_mask) == 1)  # Remove failed_mask == 0

    # Get sizes after filtering
    background_filtered_size = np.sum(background_mask)
    failed_filtered_size = np.sum(valid_mask)

    # Calculate the minimum length
    min_length = min(background_filtered_size, failed_filtered_size)

    # Check for a mismatch greater than 10%
    max_length = max(background_filtered_size, failed_filtered_size)
    if abs(background_filtered_size - failed_filtered_size) / max_length > 0.1:
        print(f"Warning: Large size mismatch! background_mask={background_filtered_size}, failed_mask={failed_filtered_size}")
        print(f"Mismatch percentage: {((background_filtered_size - failed_filtered_size) / max_length):.2f}")

    # Apply the final mask to crsData
    filtered_crsData = crsData._replace(
        diameter=np.array(crsData.diameter)[background_mask][:min_length],
        sources=np.array(crsData.sources)[background_mask][:min_length],
        sem_major=np.array(crsData.sem_major)[background_mask][:min_length],
        inc=np.array(crsData.inc)[background_mask][:min_length],
        ecc=np.array(crsData.ecc)[background_mask][:min_length],
        arg_per=np.array(crsData.arg_per)[background_mask][:min_length],
        raan=np.array(crsData.raan)[background_mask][:min_length],
        true_lat=np.array(crsData.true_lat)[background_mask][:min_length],
        background_mag=np.array(crsData.background_mag)[background_mask][:min_length],
    )

    # Apply the final mask to orbit_type_data (celmech_data)
    orbit_type_data = np.array(orbit_type_data).T

    MAX_INC = 22
    inclination_mask_crs = np.array(filtered_crsData.inc) < MAX_INC
    inclination_mask_orbit = np.array(orbit_type_data) < MAX_INC

    filtered_crsData = filtered_crsData._replace(
        diameter=filtered_crsData.diameter[inclination_mask_crs],
        sources=filtered_crsData.sources[inclination_mask_crs],
        sem_major=filtered_crsData.sem_major[inclination_mask_crs],
        inc=filtered_crsData.inc[inclination_mask_crs],
        ecc=filtered_crsData.ecc[inclination_mask_crs],
        arg_per=filtered_crsData.arg_per[inclination_mask_crs],
        raan=filtered_crsData.raan[inclination_mask_crs],
        true_lat=filtered_crsData.true_lat[inclination_mask_crs],
        background_mag=filtered_crsData.background_mag[inclination_mask_crs],
    )

    filtered_orbit_type_data = orbit_type_data[inclination_mask_orbit]

    # Ensure all datasets have the same length
    min_length = min(len(filtered_crsData.inc), len(filtered_orbit_type_data))

    filtered_crsData = filtered_crsData._replace(
        diameter=filtered_crsData.diameter[:min_length],
        sources=filtered_crsData.sources[:min_length],
        sem_major=filtered_crsData.sem_major[:min_length],
        inc=filtered_crsData.inc[:min_length],
        ecc=filtered_crsData.ecc[:min_length],
        arg_per=filtered_crsData.arg_per[:min_length],
        raan=filtered_crsData.raan[:min_length],
        true_lat=filtered_crsData.true_lat[:min_length],
        background_mag=filtered_crsData.background_mag[:min_length],
    )

    filtered_orbit_type_data = filtered_orbit_type_data[:min_length]

    celmechData = filtered_orbit_type_data
    if det: 
        filtered_crsData, celmechData = detections_filter(crsData, detData, celmechData)

    return filtered_crsData, filtered_orbit_type_data

def detections_filter(crsData: namedtuple, detData: namedtuple, celmechData: np.array): 
    """removes all objects from crossings that have not been detected. also removes them from celmech data.

    Args:
        crsData (namedtuple): crossing data
        detData (namedtuple): detections data
        celmechData (np.array): celmech data

    Returns:
        filtered_crsData (namedtuple): without crossings, detections only
        celmechData (np.array): without crossings
    """        
    # Create a mask with 1 where objects are in detData, 0 otherwise
    mask = np.isin(crsData.diameter, detData.diameter) & np.isin(crsData.sources, detData.sources)
    
    # Convert to integer mask (1 = in detData, 0 = only in crsData)
    mask = mask.astype(int)

    # Apply mask to remove objects (only keep objects where mask == 1)
    filtered_crsData = crsData._replace(
        diameter=np.array(crsData.diameter)[mask == 1],
        sources=np.array(crsData.sources)[mask == 1],
        sem_major=np.array(crsData.sem_major)[mask == 1],
        inc=np.array(crsData.inc)[mask == 1],
        ecc=np.array(crsData.ecc)[mask == 1],
        arg_per=np.array(crsData.arg_per)[mask == 1],
        raan=np.array(crsData.raan)[mask == 1],
        true_lat=np.array(crsData.true_lat)[mask == 1],
        background_mag=np.array(crsData.background_mag)[mask == 1]
    )
    #print(np.array(filtered_crsData).shape)
    #print(np.array(celmechData).shape)
    #TODO maybe this is not quite so good yet.........................................................
    # Remove objects from celmechData where mask == 0
    #filtered_celmechData = celmechData[mask == 1]

    return filtered_crsData, celmechData

def format_tle_epoch(epoch: np.array):
    """Converts epoch from Celmech in MJD to YYDDD.DDDDDDDD format for TLE. 

    Args:
        epoch (np.array): epochs from Celmech file in MJD format

    Returns:
        formatted_epochs (list): formatted epochs for TLE
    """    
    epoch = np.atleast_1d(epoch)  # Ensure array format
    jd = epoch + 2400000.5  # Convert MJD to JD
    jd_time = Time(jd, format="jd")

    formatted_epochs = []
    
    for i, t in enumerate(jd_time):
        year = t.datetime.year
        doy = t.datetime.timetuple().tm_yday
        frac_day = float(((t.datetime.hour / 24) + (t.datetime.minute / 1440) + (t.datetime.second / 86400)))
        frac_day = round(frac_day, 8)
        yy = year % 100  # Last two digits of year
        tle_epoch = f"{yy:02}{doy:03}.{str(frac_day).split('.')[1]}"
        formatted_epochs.append(tle_epoch.strip())  # Ensure no extra spaces

    return formatted_epochs if len(formatted_epochs) > 1 else formatted_epochs[0]

def compute_mean_anomaly(true_lat: np.array, arg_per: np.array, eccentricity: np.array):
    """
    Computes the mean anomaly (M) from true latitude (theta), argument of perigee (ω), and eccentricity (e).
    This function supports NumPy arrays for vectorized operations.

    Args:
        true_lat (np.array): Array of true latitudes (degrees)
        arg_per (np.array): Array of argument of perigees (degrees)
        eccentricity (np.array): Array of eccentricities

    Returns:
        np.array: Array of mean anomalies (degrees)
    """

    #compute true anomaly nu
    true_anomaly = np.radians(true_lat - arg_per)  # Convert to radians

    #compute eccentric anomaly E
    sqrt_term = np.sqrt((1 - eccentricity) / (1 + eccentricity))
    tan_E_half = sqrt_term * np.tan(true_anomaly / 2)
    eccentric_anomaly = 2 * np.arctan(tan_E_half)

    #compute mean anomaly M
    mean_anomaly_rad = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)

    mean_anomaly_deg = np.degrees(mean_anomaly_rad)

    return mean_anomaly_deg

def build_TLE(filtered_crsData: namedtuple, filtered_celmechData: namedtuple, dates: np.array, b_star_drag: np.array, output_file: str ="tle_output.txt"):
    """Generate and save properly formatted TLEs to a file.

    Args:
        filtered_crsData (namedtuple): crossing data, prefiltered and ready for TLE
        filtered_celmechData (namedtuple): celmech data, prefiltered and ready for TLE
        dates (np.array): dates from celmech file
        b_star_drag (np.array): b star drag for TLE file
        output_file (str, optional): Where to store the output file containing the TLEs. Defaults to "tle_output.txt".
    """      

    MU_EARTH = 398600.4418  # km^3/s^2
    with open(output_file, "w") as file:
        for i in tqdm(range(len(filtered_crsData.inc)), desc="Generating TLEs", unit="TLE", mininterval=0.5):
            sat_cat_no = i  + 10000 # Unused, add 10'000 to avoid leading zeros
            classification = "U"  # Unclassified
            international_designator = f"{2000 % 100:02d}{i:03d}A"  # Example: "24001A" # Unused
            epoch = format_tle_epoch(dates)[i]  # Convert to date TLE format
            decay_rate = 0.0  # Unused
            second_derivative = "00000-0"  # Unused
            b_star_drag_term = f"{b_star_drag[i]:.5e}"
            element_set_no = i  # Unused
            check1 = i  # Unused

            a = filtered_crsData.sem_major[i]
            inclination = filtered_crsData.inc[i]
            raan = filtered_crsData.raan[i]
            eccentricity = f"{filtered_crsData.ecc[i]:.7f}"[2:]  # TLE eccentricity (remove '0.')
            arg_per = filtered_crsData.arg_per[i]

            mean_anomaly = compute_mean_anomaly(filtered_crsData.true_lat[i], filtered_crsData.arg_per[i], filtered_crsData.ecc[i])

            n_rad_per_sec = np.sqrt(MU_EARTH / a**3)  # Mean motion in rad/s
            mean_motion = (n_rad_per_sec / (2 * np.pi)) * 86400  # Revolutions per day
            mean_motion = round((n_rad_per_sec / (2 * np.pi)) * 86400, 8)  # Revolutions per day, rounded
            rev_number = i  # Incrementing revolution number
            check2 = i

            line1 = (f"1 {sat_cat_no:05d}{classification} {international_designator:<8s} {epoch}  {decay_rate:.8f}  {second_derivative} {b_star_drag_term} 0  {element_set_no:04d}{check1}")
            line2 = f"2 {sat_cat_no:05d} {inclination:8.4f} {raan:8.4f} {eccentricity:7s} {arg_per:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f} {rev_number:05d} {check2}"
            file.write(line1 + "\n")
            file.write(line2 + "\n")

    print(f"TLE file '{output_file}' generated successfully.")

def prepare_input_tle(year: str, orbit_type: str, seed: int):
    """bring the input in the right format for the TLE and write the TLE. 

    Args:
        year (str): year of the data
        orbit_type (str): orbit type of the data
        seed (int): seed of the data
    """
    year2 = year[2:]
    dir = os.path.join("..", "input")

    if int(year) == 2023 and int(seed) == 1: 
        crs_filename = f"stat_Master_{year2}_{orbit_type}_s{seed}_10cm.crs"
        det_filename = f"stat_Master_{year2}_{orbit_type}_s{seed}_10cm.det"
    else: 
        crs_filename = f"stat_Master_{year2}_{orbit_type}_s{seed}.crs"
        det_filename = f"stat_Master_{year2}_{orbit_type}_s{seed}.det"

    crs_filename = os.path.join("..", "input", crs_filename)
    det_filename = os.path.join("..", "input", det_filename)

    crsData, detData = data_from_crs_and_det(crs_filename, det_filename)
    failed_mask, dates, celmech_data = data_from_celmech(year, dir, orbit_type, False, False)
    filtered_crsData, filtered_celmechData = combine_data(crsData, detData, failed_mask, dates, celmech_data)

    diameter = filtered_crsData.diameter
    semi_major = filtered_crsData.sem_major
    sources = filtered_crsData.sources
    b_star_drag, b_vals = compute_b_drag(diameter, semi_major, sources)

    build_TLE(filtered_crsData, filtered_celmechData, dates, b_star_drag)