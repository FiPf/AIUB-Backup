import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
import unique

def object_table_helper(inc_det_geo: np.array, inc_crs_geo: np.array, inc_det_gto: np.array, inc_crs_gto: np.array, inc_det_fol: np.array, inc_crs_fol: np.array):
    """returns the number of crossing objects and number of detection objects for each orbit type (GEO, GTO, followup)

    Args:
        inc_det_geo (np.array): inclinations of detection objects in GEO
        inc_crs_geo (np.array): inclinations of crossing objects in GEO
        inc_det_gto (np.array): inclinations of detection objects in GTO
        inc_crs_gto (np.array): inclinations of crossing objects in GTO
        inc_det_fol (np.array): inclinations of detection objects in followup
        inc_crs_fol (np.array): inclinations of crossing objects in followup

    Returns:
        results (np.array): contains the number of objects for detections and crossing for all orbit types
    """
    max_inc = 22
    geo_det = np.array([i for i in inc_det_geo if i < max_inc])
    geo_crs = np.array([i for i in inc_crs_geo if i < max_inc])
    gto_det = np.array([i for i in inc_det_gto if i < max_inc])
    gto_crs = np.array([i for i in inc_crs_gto if i < max_inc])
    fol_det = np.array([i for i in inc_det_fol if i < max_inc])
    fol_crs = np.array([i for i in inc_crs_fol if i < max_inc])

    results = [
        len(geo_crs), len(geo_det), len(set(geo_crs)), len(set(geo_det)),
        len(gto_crs), len(gto_det), len(set(gto_crs)), len(set(gto_det)),
        len(fol_crs), len(fol_det), len(set(fol_crs)), len(set(fol_det))
    ]

    return results

def object_table(year: str, directory: str, *obj_arrays):
    """writes a the summary of the object numbers in form of a latex table into a .*txt document

    Args:
        year (str): year, used for the description
        directory (str): place to store the *.txt file
        obj_arrays (np.arrays): contain the objects to be added to the table
    """
    obj_arrays = np.array(obj_arrays)
    
    if len(obj_arrays) == 1:
        mean = obj_arrays[0]
        std = np.zeros_like(mean)  # Assuming standard deviation as zero for single array case
    else:
        mean = np.mean(obj_arrays, axis=0)
        std = np.std(obj_arrays, axis=0, ddof=1)
    
    table_data = [
        [f"GEO {year}", f"{mean[0]:.1f} $\\pm$ {std[0]:.1f}", f"{mean[1]:.1f} $\\pm$ {std[1]:.1f}", f"{mean[2]:.1f} $\\pm$ {std[2]:.1f}", f"{mean[3]:.1f} $\\pm$ {std[3]:.1f}"],
        [f"GTO {year}", f"{mean[4]:.1f} $\\pm$ {std[4]:.1f}", f"{mean[5]:.3f} $\\pm$ {std[5]:.1f}", f"{mean[6]:.1f} $\\pm$ {std[6]:.1f}", f"{mean[7]:.1f} $\\pm$ {std[7]:.1f}"],
        [f"followup {year}", f"{mean[8]:.1f} $\\pm$ {std[8]:.1f}", f"{mean[9]:.1f} $\\pm$ {std[9]:.1f}", f"{mean[10]:.1f} $\\pm$ {std[10]:.1f}", f"{mean[11]:.1f} $\\pm$ {std[11]:.1f}"]
    ]

    table_latex = tabulate(table_data, headers=["Survey scenario", "Crossing events", "Detection events", "Unique Crossing objects", "Unique Detected objects"], tablefmt="latex_raw")
    
    table_with_caption = f"""
    \\begin{{table}}[H]
    \\centering
    \\caption{{Summary of PROOF simulation of MASTER-2009 population, {year}}}
    \\label{{tab:object_table_{year}}}
    {table_latex}
    \\end{{table}}
    """
    
    filename = f"object_table_{year}.txt"
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as output:
        output.write(table_with_caption)

def ratio_table_helper(inc_det_geo: np.array, inc_crs_geo: np.array, inc_det_gto: np.array, inc_crs_gto: np.array, inc_det_fol: np.array, inc_crs_fol: np.array):
    """calculates both ratios (1. detection events vs. crossing events and 2. detected  objects vs. crossing objects) for all orbit types (GEO, GTO, followup)

    Args:
        inc_det_geo (np.array): inclinations of detection objects in GEO
        inc_crs_geo (np.array): inclinations of crossing objects in GEO
        inc_det_gto (np.array): inclinations of detection objects in GTO
        inc_crs_gto (np.array): inclinations of crossing objects in GTO
        inc_det_fol (np.array): inclinations of detection objects in followup
        inc_crs_fol (np.array): inclinations of crossing objects in followup

    Returns:
        ratios (np.array): array containing all the detections
    """
    max_inc = 22
    geo_det = np.array([i for i in inc_det_geo if i < max_inc])
    geo_crs = np.array([i for i in inc_crs_geo if i < max_inc])
    gto_det = np.array([i for i in inc_det_gto if i < max_inc])
    gto_crs = np.array([i for i in inc_crs_gto if i < max_inc])
    fol_det = np.array([i for i in inc_det_fol if i < max_inc])
    fol_crs = np.array([i for i in inc_crs_fol if i < max_inc])
    
    geo_crossing_events = len(geo_crs)
    geo_detection_events = len(geo_det)
    geo_crossing_objects = unique.count_unique_objects_from_array(geo_crs)[0]
    geo_detected_objects = unique.count_unique_objects_from_array(geo_det)[0]

    gto_crossing_events = len(gto_crs)
    gto_detection_events = len(gto_det)
    gto_crossing_objects = unique.count_unique_objects_from_array(gto_crs)[0]
    gto_detected_objects = unique.count_unique_objects_from_array(gto_det)[0]

    followup_crossing_events = len(fol_crs)
    followup_detection_events = len(fol_det)
    followup_crossing_objects = unique.count_unique_objects_from_array(fol_crs)[0]
    followup_detected_objects = unique.count_unique_objects_from_array(fol_det)[0]

    geo_ratio_events = geo_detection_events / geo_crossing_events if geo_crossing_events != 0 else 0
    geo_ratio_objects = geo_detected_objects / geo_crossing_objects if geo_crossing_objects != 0 else 0
    gto_ratio_events = gto_detection_events / gto_crossing_events if gto_crossing_events != 0 else 0
    gto_ratio_objects = gto_detected_objects / gto_crossing_objects if gto_crossing_objects != 0 else 0
    fol_ratio_events = followup_detection_events / followup_crossing_events if followup_crossing_events != 0 else 0
    fol_ratio_objects = followup_detected_objects / followup_crossing_objects if followup_crossing_objects != 0 else 0

    ratios = [geo_ratio_events, geo_ratio_objects, gto_ratio_events, gto_ratio_objects, fol_ratio_events, fol_ratio_objects]
    
    return ratios
    
def ratio_table(year: str, directory: str, *ratio_arrays): 
    """writes a the summary of the ratios in form of a latex table into a .*txt document

    Args: 
        year (str): year, used for the title
        directory (str): place to store the *.txt file
        ratio_arrays (np.arrays): arrays containing the ratios
    """
    ratio_arrays = np.array(ratio_arrays)
    
    if len(ratio_arrays) == 1:
        mean = ratio_arrays[0]
        std = np.zeros_like(mean)  # Assuming standard deviation as zero for single array case
    else:
        mean = np.mean(ratio_arrays, axis=0)
        std = np.std(ratio_arrays, axis=0, ddof=1)
    
    table_data = [
        [f"GEO {year}", f"{mean[0]:.3f} $\\pm$ {std[0]:.3f}", f"{mean[1]:.3f} $\\pm$ {std[1]:.3f}"],
        [f"GTO {year}", f"{mean[2]:.3f} $\\pm$ {std[2]:.3f}", f"{mean[3]:.3f} $\\pm$ {std[3]:.3f}"],
        [f"followup {year}", f"{mean[4]:.3f} $\\pm$ {std[4]:.3f}", f"{mean[5]:.3f} $\\pm$ {std[5]:.3f}"]
    ]
    
    table_latex = tabulate(table_data, headers=["Survey scenario", "Crossing events vs. detection events", "Crossing objects vs. detected objects"], tablefmt="latex_raw")
    
    table_with_caption = f"""
    \\begin{{table}}[H]
    \\centering
    \\caption{{Summary of ratios from PROOF simulation of MASTER-2009 population, {year}}}
    \\label{{tab:object_table_{year}}}
    {table_latex}
    \\end{{table}}
    """
    
    filename = f"ratio_table_{year}.txt"
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as output:
        output.write(table_with_caption)
        
def write_orbele_celmech_table(data: np.array, output_file: str, year: str, orbit_type: str):
    """Write the orbital elements data into a text file in table format.

    Args:
        data (np.array): data with orbital elements
        output_file (str): file where the table should be written into
        year (str): year of the data (used for header)
        orbit_type (str): orbit type of the data (used for header)
    """
    
    output_dir = "output_celmech"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    #Title and Header
    title = f"Orbital elements from celmech, data from {year}, {orbit_type}"
    headers = ["Num Obs", "RMS", "Time Interval (s)", "Num Iter", "P", "A", "E", "I", "Node", "Perigee", "IPer"]
    
    # Calculate the number of rows based on the length of one column
    num_rows = len(data[0])
    
    # Open the file to write
    with open(output_file, 'w') as f:
        f.write(title + "\n")
        f.write(f"{' | '.join(headers)}\n")
        f.write("-" * 100 + "\n")  # Add a separator line for the table

        # Write each row of data
        for i in range(num_rows):
            row = [data[col][i] for col in range(len(headers))]  # Collect data for each column in the row
            f.write(f"{' | '.join(row)}\n")
            
def write_orbital_data_to_txt(directory: str, filename: str, objX: np.array , objY: np.array , objZ: np.array , objVx: np.array , objVy: np.array , objVz: np.array , a_vals: np.array , e_vals: np.array , i_vals: np.array , raan_vals: np.array , omega_vals: np.array , nu_vals: np.array ):
    """Write orbital data (both positions & velocities and orbital elements 1-6) into a common file

    Args:
        directory (str): where to store the final file
        filename (str): name of the final file
        objX (np.array): geocentric x coordinate
        objY (np.array): geocentric y coordinate
        objZ (np.array): geocentric z coordinate
        objVx (np.array): geocentric x velocity
        objVy (np.array): geocentric y velocity
        objVz (np.array): geocentric z velocity
        a_vals (np.array): semi major axis
        e_vals (np.array): eccentricity
        i_vals (np.array): inclination
        raan_vals (np.array): raan 
        omega_vals (np.array): omega (angle of periapsis)
        nu_vals (np.array): nu 
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)

    with open(file_path, mode='w') as file:
        file.write(f"{'X':>12} {'Y':>12} {'Z':>12} {'Vx':>12} {'Vy':>12} {'Vz':>12} {'a (km)':>12} {'e':>12} {'i (deg)':>12} {'RAAN':>12} {'omega':>12} {'nu':>12}\n")
        file.write("="*144 + "\n")  # Separator line for readability
        
        objX = objX.astype(np.float64)
        objY = objY.astype(np.float64)
        objZ = objZ.astype(np.float64)
        objVx = objVx.astype(np.float64)
        objVy = objVy.astype(np.float64)
        objVz = objVz.astype(np.float64)
        a_vals = a_vals.astype(np.float64)
        e_vals = e_vals.astype(np.float64)
        i_vals = i_vals.astype(np.float64)
        raan_vals = raan_vals.astype(np.float64)
        omega_vals = omega_vals.astype(np.float64)
        nu_vals = nu_vals.astype(np.float64)
        # Write the data rows
        
        for x, y, z, vx, vy, vz, a, e, i, raan, omega, nu in zip(objX, objY, objZ, objVx, objVy, objVz, a_vals, e_vals, i_vals, raan_vals, omega_vals, nu_vals):
            #print(f"Data types: x: {type(x)}, y: {type(y)}, z: {type(z)}, vx: {type(vx)}, vy: {type(vy)}, vz: {type(vz)}, "
                #f"a: {type(a)}, e: {type(e)}, i: {type(i)}, raan: {type(raan)}, omega: {type(omega)}, nu: {type(nu)}")
            
            file.write(f"{x:12.6f} {y:12.6f} {z:12.6f} {vx:12.6f} {vy:12.6f} {vz:12.6f} {a:12.6f} {e:12.6f} {i:12.6f} {raan:12.6f} {omega:12.6f} {nu:12.6f}\n")
            
    print(f"Data written to {filename} successfully.")