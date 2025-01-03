import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
from datetime import datetime, timedelta
import getdata
import tables

def sources_vs_sizes(sources: np.array, diameter: np.array): 
    """calculate the average sizes per source and print them to the screen

    Args:
        sources (np.array): sources of the objects
        diameter (np.array): diameters of the objects
    """
    ind_1 = [i for i in range(len(sources)) if sources[i] == 0+1]
    ind_2 = [i for i in range(len(sources)) if sources[i] == 1+1]
    ind_3 = [i for i in range(len(sources)) if sources[i] == 2+1]
    ind_4 = [i for i in range(len(sources)) if sources[i] == 3+1]
    ind_5 = [i for i in range(len(sources)) if sources[i] == 4+1]
    ind_6 = [i for i in range(len(sources)) if sources[i] == 5+1]
    
    d1 = [diameter[i] for i in ind_1]
    print(d1)
    d2 = [diameter[i] for i in ind_2]
    print(d2)
    d3 = [diameter[i] for i in ind_3]
    print(d3)
    d4 = [diameter[i] for i in ind_4]
    print(d4)
    d5 = [diameter[i] for i in ind_5]
    print(d5)
    d6 = [diameter[i] for i in ind_6]
    print(d6)
    
    diameter_1 = np.mean([diameter[i] for i in ind_1])
    diameter_2 = np.mean([diameter[i] for i in ind_2])
    diameter_3 = np.mean([diameter[i] for i in ind_3])
    diameter_4 = np.mean([diameter[i] for i in ind_4])
    diameter_5 = np.mean([diameter[i] for i in ind_5])
    diameter_6 = np.mean([diameter[i] for i in ind_6])
    
    print(f"Average Diameter 1 (Fragments): {diameter_1:.3f}")
    print(f"Average Diameter 2 (SRM slag): {diameter_2:.3f}")
    print(f"Average Diameter 3 (NaK droplets): {diameter_3:.3f}")
    print(f"Average Diameter 4 (TLEs): {diameter_4:.3f}")
    print(f"Average Diameter 5 (Westford Needles): {diameter_5:.3f}")
    print(f"Average Diameter 6 (Multi-Layered Insulation): {diameter_6:.3f}")
    
def corrected_ratio(inc_det: np.array, inc_crs: np.array): 
    """calculates the new ratio (detected vs. crossing) excluding all objects with inclination higher than i = 40°

    Args:
        inc_det (np.array): inclinations of detection objects
        inc_crs (np.array): inclinations of crossing objects
    """
    max_inc = 22 
    new_inc_det = np.array([i for i in inc_det if i < max_inc])
    new_inc_crs = np.array([i for i in inc_crs if i < max_inc])
    
    ratio = len(new_inc_det)/len(new_inc_crs)
    print(f"The corrected ratio of detected objects vs. crossing objects: {ratio:.3f}")
    
    #print(f"Number of corrected detection events: {len(new_inc_det)}")
    #print(f"Number of corrected crossing events: {len(new_inc_crs)}")

def integration_time_counter(aut_files: np.array):
    """Gets the sum of all integration times in multiple .aut files.

    Args:
        aut_files (np.array): Names of the .aut files to search for within the 'aut_files' directory.

    Returns:
        total_integration_time (float): Total integration time for all found .aut files.
    """
    total_integration_time = 0.0
    aut_files_directory = "aut_files"  # Base directory where the search begins

    # Traverse the directory tree
    for root, _, files in os.walk(aut_files_directory):
        for file_name in aut_files:
            if file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as inp:
                    data = inp.readlines()

                    integration_time = []
                    gap_time = []
                    duration = []
                    
                    for line in data[2:]:  # Assuming integration times start from the third line
                        parts = line.split()
                        if len(parts) > 12:  # Ensure the line has enough parts to avoid IndexError
                            try:
                                integration_time.append(float(parts[11]))  # Convert to float
                                duration.append(float(parts[5]))
                                gap_time.append(float(parts[12]))
                            except ValueError:
                                continue  # Skip lines with invalid data

                    # Convert lists to numpy arrays for calculations
                    integration_time = np.array(integration_time)
                    gap_time = np.array(gap_time)
                    duration = np.array(duration)
                    duration = 60*60*duration
                    
                    # Calculate the number of observations
                    num_obs = [d / (i + g) for d, i, g in zip(duration, integration_time, gap_time)]
                    total_num_of_images = np.sum(num_obs)
                    total_integration_time += total_num_of_images

    return total_integration_time

def mjd_to_date(mjd):
    """
    Convert Modified Julian Date (MJD) to a standard date format.
    
    Parameters:
    mjd (float): Modified Julian Date to be converted.
    
    Returns:
    str: Date in the format YYYY-MM-DD HH:MM:SS
    """
    # MJD reference date: 17 November 1858
    mjd_ref_date = datetime(1858, 11, 17)
    # Convert MJD to date
    date = mjd_ref_date + timedelta(days=mjd)
    return date.strftime('%Y-%m-%d %H:%M:%S')
    
def date_to_mjd_manual(date_string):
    """
    Convert a date and time to Modified Julian Date (MJD) manually without Astropy.
    
    Parameters:
        date_string (str): A single string containing date and time in the format
                           'YYYY MM DD HH MM SS.SSS' (e.g., '2005 1 4 20 48 58.000').
    
    Returns:
        float: The corresponding Modified Julian Date (MJD).
    """
    # Parse the date string
    year, month, day, hour, minute, second = map(float, date_string.split())

    # Adjust months and years if the month is January or February
    if month <= 2:
        year -= 1
        month += 12

    # Calculate the Julian Date (JD)
    A = int(year // 100)
    B = 2 - A + (A // 4)  # Gregorian calendar adjustment
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    JD += (hour / 24.0) + (minute / 1440.0) + (second / 86400.0)

    # Convert JD to MJD
    MJD = JD - 2400000.5
    return MJD


def calculate_orbital_elements(r: np.array, v: np.array, mu: float = 3.986004418e14):
    """Calculate orbital elements from given position and velocity vectors. 
    Both vectors need to be in a geocentric coordinate system!

    Args:
        r (np.array): 3D position vector
        v (np.array): 3D velocity vector
        mu (float): Gravitational G*M of Earth

    Returns:
        a (float): Semi-major axis
        e (float): Eccentricity
        i (float): Inclination
        Omega (float): RAAN
        omega (float): Argument of periapsis
        nu (float): True anomaly
    """
    r = r * 1000  # Convert from kilometers to meters
    v = v * 1000  # Convert from kilometers to meters

    r0 = np.linalg.norm(r)  # Magnitude of position vector
    v0 = np.linalg.norm(v)  # Magnitude of velocity vector

    h = np.cross(r, v)  # Specific angular momentum vector
    h_mag = np.linalg.norm(h)  # Magnitude of the angular momentum vector

    p = h_mag**2 / mu  # Semi-latus rectum
    e_vector = (np.cross(v, h) / mu) - (r / r0)  # Eccentricity vector

    e = np.linalg.norm(e_vector)  # Magnitude of eccentricity vector
    a = p / (1 - e**2)  # Semi-major axis


    i = np.arccos(h[2] / h_mag)  # Inclination
    
    k = np.array([0, 0, 1])
    n = np.cross(k, h)
    Omega = np.arctan2(n[1], n[0])# RAAN
    
    omega = np.arctan2(e_vector[2], np.dot(e_vector[:2], np.cross([0, 0, 1], h)[:2]))  # Argument of periapsis
    nu = np.arctan2(np.dot(e_vector, r), np.dot(e_vector, r))  # True anomaly

    if np.degrees(i) > 22:
        return None

    return a, e, np.degrees(i), np.degrees(Omega), np.degrees(omega), np.degrees(nu)


def from_plugin_to_orbital_elements(file_path: str, year: str, orbit_type: str): 
    """iterate through all objects from the plugin *.pro file and calculate the orbital elements using the 
    calculate_orbital_elements function. write the calculated orbital elements (together with positions and velocities from plugin)
    into a text file

    Args:
        file_path (str): *.pro file where the positions and velocities are stored
        year (str): year of the data (used for file header)
        orbit_type (str): orbit type of the data (used for file header)
        """
    
    data = getdata.array_extender_plugin(file_path)
    ifile = data[1]

    unique_i_file, unique_i_file_indexes = np.unique(ifile, return_index=True)
    
    
    objX = data[6]
    objY = data[7]
    objZ = data[8]
    objVx = data[9]
    objVy = data[10]
    objVz = data[11]

    objX = np.array(data[6], dtype=float)[unique_i_file_indexes]
    objY = np.array(data[7], dtype=float)[unique_i_file_indexes]
    objZ = np.array(data[8], dtype=float)[unique_i_file_indexes]
    objVx = np.array(data[9], dtype=float)[unique_i_file_indexes]
    objVy = np.array(data[10], dtype=float)[unique_i_file_indexes]
    objVz = np.array(data[11], dtype=float)[unique_i_file_indexes]
    
    a_vals = []
    e_vals = []
    i_vals = []
    raan_vals = []
    omega_vals = []
    nu_vals = []
    for x, y, z, vx, vy, vz in zip(objX, objY, objZ, objVx, objVy, objVz):
        pos = np.array([x,y,z])
        vel = np.array([vx, vy, vz])
        
        result = calculate_orbital_elements(pos, vel)
        if result is None:
            continue  # Skip to the next iteration

        # Unpack and store the valid orbital elements
        a, e, i, raan, omega, nu = result
        a_vals.append(a)
        e_vals.append(e)
        i_vals.append(i)
        raan_vals.append(raan)
        omega_vals.append(omega)
        nu_vals.append(nu)
        a_vals.append(a)
        e_vals.append(e)
        i_vals.append(i)
        raan_vals.append(raan)
        omega_vals.append(omega)
        nu_vals.append(nu)
        
    dir = os.path.join("output_celmech", "txt_files")
    name = f"orbital_elements_from_plugin_{year}_{orbit_type}"
    
    objX = np.array(objX,  dtype = float)
    objY = np.array(objY,  dtype = float)
    objZ = np.array(objZ,  dtype = float)
    objVx = np.array(objVx,  dtype = float)
    objVy = np.array(objVy,  dtype = float)
    objVz = np.array(objVz,  dtype = float)
    a_vals = np.array(a_vals, dtype = float)
    e_vals = np.array(e_vals,  dtype = float)
    i_vals = np.array(i_vals,  dtype = float)
    raan_vals = np.array(raan_vals,  dtype = float)
    omega_vals = np.array(omega_vals,  dtype = float)
    nu_vals = np.array(nu_vals,  dtype = float)
    
    tables.write_orbital_data_to_txt(dir, name, objX, objY, objZ, objVx, objVy, objVz, a_vals, e_vals, i_vals, raan_vals, omega_vals, nu_vals)
