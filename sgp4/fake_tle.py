#this file contains all functions to generate a fake tle, which is used for the input of the sgp4 propagator
#it takes the epoch from the plugin.pro file and the orbital elements from the *.crs file
import numpy as np
import getdata
from getdata import PopulationType, data_returner
from collections import namedtuple
import os

def compute_b_drag(diameter: np.array, semi_major: np.array, sources: np.array):
    b_star_drag_vals = []
    area_to_mass_vals = []
    for d, s in zip(diameter, sources): 
        a_m = compute_am(d, s)
        area_to_mass_vals.append(a_m)

    drag_coeff = 2.2
    density_const = 0.15696615
    R_earth = 6378

    for ele, a in zip(area_to_mass_vals, semi_major): 
        rho = density_const
        b_star_drag = drag_coeff*ele*rho*0.5
        b_star_drag_vals.append(b_star_drag)

    return b_star_drag_vals 

def compute_am(d: float, source: int):
    """
    Compute the area-to-mass ratio (A/m) for a given object size d.
    :param d: Object size (diameter) in meters
    :param object_type: 'rocket' for rocket bodies, 'spacecraft' for payloads
    :return: Area-to-mass ratio A/m in m^2/kg
    """
    object_type = ""

    if source == 1: 
        object_type == 'spacecraft'
    if source == 4: 
        object_type == 'rocket'
    else: 
        object_type == 'None' #small objects only, all sources excpet 1 (=fragments) and 4 (=TLE)

    log_d = np.log10(d)
    
    # Define bridging function threshold only 
    if object_type != 'None': 
        if object_type == 'rocket':
            r_thresh = 10 ** (log_d + 1.76)
            small_particle_limit = 1.7  # cm
        elif object_type == 'spacecraft':
            r_thresh = 10 ** (log_d + 1.05)
            small_particle_limit = 8  # cm
    
    # Generate random r for decision making
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

def data_from_crs(crs_filename: str): 

    crsData = namedtuple("crsData", ["diameter", "sources", "sem_major", "inc", "ecc", "arg_per", "raan", "true_lat", "background_mag"])

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

    return crsData(diameter, sources, sem_major, inc, ecc, arg_per, raan, true_lat, background_mag)

def data_from_plugin(year: str, orbit_type: str):
    pluginData = namedtuple("pluginData", ["epoch"])
    err = False
    ell = False
    filename = getdata.get_celmech_OUT_files(year, orbit_type, err, ell) #find the right Celmech file for given year and orbit type
    data = getdata.array_extender_plugin(filename)
    epoch = data[2]
    return pluginData(epoch) 

def data_from_celmech(year:str, dir: str, err:bool, ell: bool, orbit_type: str): 
    population_type = PopulationType.NORMAL
    total_num_of_objects = 0
    yy = year[2:]

    orbit_type_data_dict = {"num_obs": [], "rms": [], "time_interval": [], "num_iter": [], 
                "P": [], "A": [], "E": [], "I": [], "Node": [], "Per": [], "IPer": []}

    files = []
    file = getdata.get_celmech_OUT_files(year, orbit_type, err, ell) #find the right Celmech file for given year and orbit_type type
    file = os.path.join("input_celmech", file)
    files.append(file)
    orbit_type_data, number_of_obj, dates, failed_mask = getdata.get_orbele_and_date_from_celmech(files) #extract orbit_typeal elements from Celmech files
    
    total_num_of_objects += number_of_obj
    
    # Append the data to the corresponding orbit_type type in the dictionary
    orbit_type_data_dict[orbit_type]["num_obs"].append(orbit_type_data[0])
    orbit_type_data_dict[orbit_type]["rms"].append(orbit_type_data[1])
    orbit_type_data_dict[orbit_type]["time_interval"].append(orbit_type_data[2])
    orbit_type_data_dict[orbit_type]["num_iter"].append(orbit_type_data[3])
    orbit_type_data_dict[orbit_type]["P"].append(orbit_type_data[4])
    orbit_type_data_dict[orbit_type]["A"].append(orbit_type_data[5])
    orbit_type_data_dict[orbit_type]["E"].append(orbit_type_data[6])
    orbit_type_data_dict[orbit_type]["I"].append(orbit_type_data[7])
    orbit_type_data_dict[orbit_type]["Node"].append(orbit_type_data[8])
    orbit_type_data_dict[orbit_type]["Per"].append(orbit_type_data[9])
    orbit_type_data_dict[orbit_type]["IPer"].append(orbit_type_data[10])            
    
    maxinc = 22 #cut off all data at greater inclinations
    apogee_threshold = 10000  # in kilometers (10k km)

    # Filtering for inclination and apogee
    inc_data = np.array(orbit_type_data_dict[orbit_type]["I"])
    node_data = np.array(orbit_type_data_dict[orbit_type]["Node"])
    ecc_data = np.array(orbit_type_data_dict[orbit_type]["E"])
    a_data = np.array(orbit_type_data_dict[orbit_type]["A"])

    mask = inc_data <= maxinc
    orbit_type_data_dict[orbit_type]["I"] = inc_data[mask]
    orbit_type_data_dict[orbit_type]["Node"] = node_data[mask]
    orbit_type_data_dict[orbit_type]["E"] = ecc_data[mask]
    orbit_type_data_dict[orbit_type]["A"] = a_data[mask]

    return failed_mask

def combine_data(crsData: namedtuple, pluginData: namedtuple, failed_mask: np.array):
    # Convert to NumPy arrays
    background_mask = np.array(crsData.background_mag) != 0  # Remove background_mag = 0
    valid_mask = (np.array(failed_mask) == 1) & background_mask  # Remove failed_mask == 0

    # Apply the final mask to crsData
    filtered_crsData = crsData._replace(
        diameter=np.array(crsData.diameter)[valid_mask],
        sources=np.array(crsData.sources)[valid_mask],
        sem_major=np.array(crsData.sem_major)[valid_mask],
        inc=np.array(crsData.inc)[valid_mask],
        ecc=np.array(crsData.ecc)[valid_mask],
        arg_per=np.array(crsData.arg_per)[valid_mask],
        raan=np.array(crsData.raan)[valid_mask],
        true_lat=np.array(crsData.true_lat)[valid_mask],
        background_mag=np.array(crsData.background_mag)[valid_mask],
    )

    # Apply the final mask to pluginData
    filtered_pluginData = pluginData._replace(
        epoch=np.array(pluginData.epoch)[valid_mask]
    )

    MAX_INC =22
    inclination_mask_crs = np.array(filtered_crsData.inc) < MAX_INC
    inclination_mask_plugin = np.array(filtered_pluginData.inc) < MAX_INC

    filtered_crsData = crsData._replace(
        diameter=np.array(crsData.diameter)[inclination_mask_crs],
        sources=np.array(crsData.sources)[inclination_mask_crs],
        sem_major=np.array(crsData.sem_major)[inclination_mask_crs],
        inc=np.array(crsData.inc)[inclination_mask_crs],
        ecc=np.array(crsData.ecc)[inclination_mask_crs],
        arg_per=np.array(crsData.arg_per)[inclination_mask_crs],
        raan=np.array(crsData.raan)[inclination_mask_crs],
        true_lat=np.array(crsData.true_lat)[inclination_mask_crs],
        background_mag=np.array(crsData.background_mag)[inclination_mask_crs],
    )

    filtered_pluginData = pluginData._replace(
        epoch=np.array(pluginData.epoch)[inclination_mask_plugin]
    )

    min_length = min(len(filtered_crsData.inc), len(filtered_pluginData.epoch))

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

    filtered_pluginData = filtered_pluginData._replace(
        epoch=filtered_pluginData.epoch[:min_length]
    )

    return filtered_crsData, filtered_pluginData

#TODOOOOOO
def detections_filter(): 
    pass

def format_tle_epoch(epoch):
    """Convert epoch (YYYYMMDD.ddd) into YYDDD.DDDDDDDD format for TLE."""
    year = int(str(epoch)[:4])  # Extract year
    day_of_year = int(str(epoch)[4:7])  # Extract day of year
    decimal_part = float("0." + str(epoch)[7:])  # Extract decimal part
    yy = year % 100  # Convert to two-digit year
    return f"{yy:02}{day_of_year:03}{decimal_part:.8f}"  # Format as YYDDD.DDDDDDDD

def build_TLE(filtered_crsData: namedtuple, filtered_pluginData: namedtuple, b_star_drag: np.array, output_file="tle_output.txt"):
    """Generate and save properly formatted TLEs to a file."""
    
    MU_EARTH = 398600.4418  # km^3/s^2
    with open(output_file, "w") as file:
        for i in range(len(filtered_crsData.inc)): #loop through all objects
            sat_cat_no = i #unused
            classification = "U"  # Unclassified
            international_designator = i  #unused
            epoch = format_tle_epoch(filtered_pluginData.epoch[i])  # convert to date TLE format
            decay_rate = 0.0 #unused
            second_derivative = "00000-0" #unused
            b_star_drag_term = f"{b_star_drag[i]:.5e}"
            element_set_no = i #unused
            check1 = i #unused

            a = filtered_crsData.sem_maj[i]
            inclination = filtered_crsData.inc[i]
            raan = filtered_crsData.raan[i]
            eccentricity = f"{filtered_crsData.ecc[i]:.7f}"[2:]  # TLE eccentricity (remove '0.')
            arg_per = filtered_crsData.arg_per[i]
            mean_anomaly = filtered_crsData.true_lat[i]  # Approximating as true latitude
            n_rad_per_sec = np.sqrt(MU_EARTH / a**3)  # Mean motion in rad/s
            mean_motion = (n_rad_per_sec / (2 * np.pi)) * 86400 #revolutions per day
            rev_number = i  # Incrementing revolution number
            check2 = i

            line1 = f"1 {sat_cat_no:05d}{classification} {international_designator}   {epoch}  {decay_rate:.8f}  {second_derivative} {b_star_drag_term} 0  {element_set_no:04d}{check1}"
            line2 = f"2 {sat_cat_no:05d}  {inclination:8.4f} {raan:8.4f} {eccentricity:7s}  {arg_per:8.4f}  {mean_anomaly:8.4f} {mean_motion:11.8f}{rev_number:05d}{check2}"
            file.write(line1 + "\n")
            file.write(line2 + "\n")

    print(f"TLE file '{output_file}' generated successfully.")