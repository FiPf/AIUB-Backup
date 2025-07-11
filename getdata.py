import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
import sortdata
import plotting
import calculations
from enum import Enum
import matplotlib.cm as cm
from collections import Counter
from matplotlib.patches import Patch


def clear_directory(directory: str): 
    """delete every file from the current directory (used to ensure that no plots/files are overwritten when rerunning the code)

    Args:
        directory (str): directory to clear
    """
    files = os.listdir(directory)
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
                os.remove(file_path)

def array_extender(filename: str): 
    """function to store the data from the *.det and *.crs in numpy arrays.
    Args:
        filename (str): *.crs or *.det file 

    Returns:
        data (np.array): np.array of np.arrays containing all the data
    """
    with open(filename, "r") as inp: 
        data = inp.readlines()
    
    ID = []
    diameter = []
    factor = []
    source = []
    TCA = []
    TCA_RNG = []
    TCA_ALT = []
    TCA_RRT = []
    sem_major = []
    inc = []
    ecc = []
    arg_per = []
    raan = []
    true_lat = []
    fov_dwell = []
    ang_vel = []
    pathoffs = []
    albedo = []
    phs_ang = []
    illumination = []
    mag_obj = []
    mag_backgr = []
    max_snr = []
    RA_LOS = []
    Des_LOS = []
    
    for line in data[22:]: 
        if '***' in line:
            continue
        parts = line.split()
        if len(parts) >= 25 and parts[0].replace('.', '', 1).isdigit():
        # Ensure the line has enough elements, removes the last few lines independent of file length
            ID.append(float(parts[0]))
            diameter.append(float(parts[1]))
            factor.append(float(parts[2]))
            source.append(float(parts[3]))  
            TCA.append(float(parts[4]))
            TCA_RNG.append(float(parts[5]))
            TCA_ALT.append(float(parts[6]))
            TCA_RRT.append(float(parts[7]))
            sem_major.append(float(parts[8]))
            inc.append(float(parts[9]))
            ecc.append(float(parts[10]))
            arg_per.append(float(parts[11]))
            raan.append(float(parts[12]))
            true_lat.append(float(parts[13]))
            fov_dwell.append(float(parts[14]))
            ang_vel.append(float(parts[15]))
            pathoffs.append(float(parts[16]))
            albedo.append(float(parts[17]))
            phs_ang.append(float(parts[18]))
            illumination.append(float(parts[19]))
            mag_obj.append(float(parts[20]))
            mag_backgr.append(float(parts[21]))
            max_snr.append(float(parts[22]))
            RA_LOS.append(float(parts[23]))
            Des_LOS.append(float(parts[24]))

    data1 = [np.array(ID), np.array(diameter), np.array(factor), np.array(source), np.array(TCA), np.array(TCA_RNG), np.array(TCA_ALT), np.array(TCA_RRT), np.array(sem_major)]
    data2 = [np.array(inc), np.array(ecc), np.array(arg_per), np.array(raan), np.array(true_lat), np.array(fov_dwell), np.array(ang_vel), np.array(pathoffs), np.array(albedo)]
    data3 = [np.array(phs_ang), np.array(illumination), np.array(mag_obj), np.array(mag_backgr), np.array(max_snr), np.array(RA_LOS), np.array(Des_LOS)]
    return data1 + data2 + data3

def array_extender_obs(filenames: list): 
    """takes multiple observation files and returns the data contained in it as an array

    Args:
        filenames (list): list of filenames

    Returns:
        data (np.array): data from all files in filenames
    """

    campaign = []
    name = []
    osc_epoch = []
    date = []
    time = []
    arcl = []
    num_obs = []
    mag_c = []
    mag_u = []
    sem_maj = []
    ecc = []
    inc = []
    raan = []
    w_peri = []
    mean_anomaly = []
    dn = []
    da = []
    di = []
    draan = []
    dw = []
    ds = []
    RA_h = []
    long = []
    lat = []
    
    for file in filenames: 
        with open(file, "r") as inp: 
            lines = inp.readlines()
            for line in lines[1:]:
                parts = line.split()
                
                if len(parts) >= 22: # Ensure the line has enough elements
                    campaign.append(parts[0])
                    name.append(parts[1])
                    osc_epoch.append(parts[2])
                    date.append(parts[3])
                    time.append(parts[4])
                    arcl.append(float(parts[5]))
                    num_obs.append(float(parts[6]))
                    mag_c.append(float(parts[7]))
                    mag_u.append(float(parts[8]))
                    sem_maj.append(float(parts[9]))
                    ecc.append(float(parts[10]))
                    inc.append(float(parts[11]))
                    raan.append(float(parts[12]))
                    w_peri.append(float(parts[13]))
                    mean_anomaly.append(float(parts[14]))
                    dn.append(float(parts[15]))
                    da.append(float(parts[16]))
                    di.append(float(parts[17]))
                    draan.append(float(parts[18]))
                    dw.append(float(parts[19]))
                    ds.append(float(parts[20]))
                    RA_h.append(float(parts[21]))
                    long.append(float(parts[22]))
                    lat.append(float(parts[23]))
                    
    data = [
        np.array(campaign), np.array(name), np.array(osc_epoch), np.array(date), np.array(time),
        np.array(arcl), np.array(num_obs), np.array(mag_c), np.array(mag_u), np.array(sem_maj),
        np.array(ecc), np.array(inc), np.array(raan), np.array(w_peri), np.array(mean_anomaly),
        np.array(dn), np.array(da), np.array(di), np.array(draan), np.array(dw), np.array(ds),
        np.array(RA_h), np.array(long), np.array(lat)
    ]
    
    return data

def array_extender_6param(filenames: list):
    object = []
    alt_name = []
    epoch_i = []
    arc = []
    nobs_i = []
    semi_major_i = []
    ecc_i = []
    inc_i = []
    raan_i = []
    w_i = []
    M_i = []
    osc_epoch = []
    semi_major_osc = []
    ecc_osc = []
    inc_osc = []
    raan_osc = []
    w_osc = []
    M_osc = []
    da = []
    de = []
    di = []
    draan = []
    dw = []
    dM = []
    AM = []
    dAM = []
    mag = []
    smag = []
    mag_pc = []
    smag_pc = []
    survey = []
    
    for file in filenames: 
        with open(file, "r") as inp: 
            lines = inp.readlines()
            
            for line in lines[1:]: 
                parts = line.split()
                
                try:
                    semi_major_value = float(parts[5])
                    ecc = float(parts[6])
                    inc = float(parts[7])
                    nod = float(parts[8])
                except:
                    continue
                
                if float(parts[6]) > 1: 
                    continue
                    
                if len(parts) >= 24:  # Ensure the line has enough elements
                    object.append(parts[0])
                    alt_name.append(parts[1])
                    epoch_i.append(parts[2])
                    arc.append(parts[3])
                    nobs_i.append(parts[4])
                    semi_major_i.append(float(parts[5]))
                    ecc_i.append(float(parts[6]))
                    inc_i.append(float(parts[7]))
                    raan_i.append(float(parts[8]))
                    w_i.append(float(parts[9]))
                    M_i.append(float(parts[10]))
                    osc_epoch.append(float(parts[11]))
                    semi_major_osc.append(float(parts[12]))
                    ecc_osc.append(float(parts[13]))
                    inc_osc.append(float(parts[14]))
                    raan_osc.append(float(parts[15]))
                    w_osc.append(float(parts[16]))
                    M_osc.append(float(parts[17]))
                    da.append(float(parts[18]))
                    de.append(float(parts[18]))
                    di.append(float(parts[20]))
                    draan.append(float(parts[21]))
                    dw.append(float(parts[22]))
                    dM.append(float(parts[23]))
                    AM.append(float(parts[24]))
                    dAM.append(parts[25])
                    mag.append(float(parts[26]))
                    smag.append(float(parts[27]))
                    mag_pc.append(float(parts[28]))
                    smag_pc.append(float(parts[29]))
                    survey.append(parts[30])
                    
    data = [np.array(object), np.array(alt_name), np.array(epoch_i), np.array(arc), np.array(nobs_i), 
            np.array(semi_major_i), np.array(ecc_i), np.array(inc_i), np.array(raan_i), np.array(w_i), 
            np.array(M_i), np.array(osc_epoch), np.array(semi_major_osc), np.array(ecc_osc), np.array(inc_osc), 
            np.array(raan_osc), np.array(w_osc), np.array(M_osc), np.array(da), np.array(de), np.array(di), 
            np.array(draan), np.array(dw), np.array(dM), np.array(AM), np.array(dAM), np.array(mag), 
            np.array(smag), np.array(mag_pc), np.array(smag_pc), np.array(survey)]
    return data

def extract_separated_obs_files(year: str, dir: str):
    #look in the directory "sorted_observation_files"
    #open the folder with year as title
    #call array extender obs on every file inside this folder
    #return data of each array
    #dir is either "sorted_observation_files" (uncorrelated) or "sorted_corr_observation_files" (correlated)
    
    directory = os.path.join(dir, year)
    filenames = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    if len(filenames) != 3: 
        raise ValueError("Something is not good, there should be three files per year (GEO; GTO; fol)")
    
    GEO_data = filenames[0]
    GTO_data = filenames[1]
    fol_data = filenames[2]
        
    if "corr" not in dir: 
        GEO_data = array_extender_obs([GEO_data])
        GTO_data = array_extender_obs([GTO_data])
        fol_data = array_extender_obs([fol_data])
    else: 
        GEO_data = array_extender_correlated_obs([GEO_data])
        GTO_data = array_extender_correlated_obs([GTO_data])
        fol_data = array_extender_correlated_obs([fol_data])
        
    return GEO_data, GTO_data, fol_data

def array_extender_correlated_obs(filenames: list):
    """Takes multiple observation files and returns the data contained in them as arrays.

    Args:
        filenames (list): List of filenames.

    Returns:
        data (list of np.array): Data from all files in filenames, each column stored as a separate NumPy array.
    """

    # Initialize lists to store data for each column
    campaign = []
    name = []
    osc_epoch = []
    date = []
    time = []
    arcl = []
    num_obs = []
    mag_c = []
    mag_u = []
    sem_maj = []
    ecc = []
    inc = []
    raan = []
    w_peri = []
    mean_anomaly = []
    dn = []
    da = []
    di = []
    draan = []
    dw = []
    ds = []
    RA_h = []
    decl = []
    lon = []
    lat = []

    for file in filenames:
        with open(file, "r") as inp:
            lines = inp.readlines()
            for line in lines[1:]:
                # Clean up the line and use fixed-width slicing to extract each column
                line = line.strip()

                # Extract data using fixed-width positions based on sample data
                if len(line) >= 129:  # Ensure the line is long enough
                    campaign.append(line[0:16].strip())
                    name.append(line[16:53].strip())
                    osc_epoch.append(line[53:65].strip())
                    date.append(line[65:74].strip())
                    time.append(line[74:87].strip())
                    arcl.append(float(line[90:104].strip()))
                    num_obs.append(int(line[104:109].strip()))
                    mag_c.append(float(line[109:116].strip())) #7
                    mag_u.append(float(line[116:123].strip())) #8
                    sem_maj.append(float(line[123:136].strip())) #9
                    ecc.append(float(line[136:148].strip()))#10
                    inc.append(float(line[148:157].strip()))#11
                    raan.append(float(line[157:170].strip()))
                    w_peri.append(float(line[170:179].strip()))
                    mean_anomaly.append(float(line[179:190].strip()))
                    dn.append(float(line[190:202].strip()))
                    da.append(float(line[202:212].strip()))
                    di.append(float(line[212:222].strip()))
                    draan.append(float(line[222:230].strip()))
                    dw.append(line[230:235].strip())
                    ds.append(float(line[235:241].strip()))
                    RA_h.append(float(line[241:247].strip()))
                    decl.append(float(line[247:254].strip()))
                    lon.append(line[254:259].strip())
                    lat.append(float(line[259:266].strip()))

    # Convert lists to NumPy arrays
    data = [
        np.array(campaign), np.array(name), np.array(osc_epoch), np.array(date), np.array(time),
        np.array(arcl), np.array(num_obs), np.array(mag_c), np.array(mag_u), np.array(sem_maj),
        np.array(ecc), np.array(inc), np.array(raan), np.array(w_peri), np.array(mean_anomaly),
        np.array(dn), np.array(da), np.array(di), np.array(draan), np.array(dw), np.array(ds),
        np.array(RA_h), np.array(decl), np.array(lon)
    ]
    
    return data


def array_extender_plugin(filename: str): 
    runid = []
    ifile = []
    epoch = []
    topoX = []
    topoY = []
    topoZ = []
    objX = []
    objY = []
    objZ = []
    objVx = []
    objVy = []
    objVz = []
    objV = []
    objID = []
    
    with open(filename, "r") as inp: 
        lines = inp.readlines()
        for line in lines[1:]: 
            parts = line.split()
            runid.append(parts[0])
            ifile.append(parts[1])
            epoch.append(parts[2])
            topoX.append(parts[3])
            topoY.append(parts[4])
            topoZ.append(parts[5])
            objX.append(parts[6])
            objY.append(parts[7])
            objZ.append(parts[8])
            objVx.append(parts[9])
            objVy.append(parts[10])
            objVz.append(parts[11])
            objV.append(parts[12])
            objID.append(parts[13])
            
    data = [runid, ifile, epoch, topoX, topoY, topoZ, objX, objY, objZ, objVx, objVy, objVz, objV, objID]
    data = np.array(data)
    return data

def crossing_minus_detected(crs_data: np.array, det_data: np.array): 
    """find the objects that are in crossings, but not in detections

    Args:
        crs_file (str): file with the crossings
        det_file (str): file with the detections

    Returns:
        crs_minus_det (np.array): objects that are in crossings and not in detections
    """
    crs_minus_det = []
        
    crs_id = crs_data[0]
    det_id = det_data[0]
        
    det_id_set = set(det_id)
    
    for i, id in enumerate(crs_id):
        if id not in det_id_set:
            crs_minus_det.append(crs_data[:,i])
    
    crs_minus_det = np.array(crs_minus_det)
    
    print(crs_minus_det.shape)

    return crs_minus_det

def get_celmech_OUT_files(year: str, orbit_type: str, err: bool, ell: bool):
    """Get the right celmech OUT file from the directory where these files are stored.

    Args:
        year (str): Year of the data
        orbit_type (str): Orbit type of the data
        err (bool): If True, the file must contain 'err'.
        ell (bool): If True, the file must contain "ell" and cannot contain "err" (such files should not exist).

    Raises:
        ValueError: When no file with the specified year, orbit type, or conditions is found.

    Returns:
        filename (str): Name of the file as string.
    """
    directory = os.path.join("..", "input_celmech")
    print(year[2:], orbit_type)

    for filename in os.listdir(directory):
        # Check if both the year and orbit_type are in the filename
        if year[2:] in filename and orbit_type in filename:
            # If ell is True, return the file containing 'ell' if it exists
            if ell and "ell" in filename:
                return filename
            # If err is True, the filename should contain 'err'
            elif err and "err" in filename:
                return filename
            # If err is False, the filename should NOT contain 'err'
            elif not err and "err" not in filename and not ell and not "ell" in filename:
                return filename

    raise ValueError("No OUT file found for the specified year, orbit type, and error condition!")

def get_orbele_from_celmech(OUT_file_list: list):
    """open the OUT files from a list of files. extracts orbital elements and stores them in numpy array.
    Function can be used for both circular and elliptical orbits (from Celmech OUT files). 
    That is why there are two target line options. 
    
    Args:
        OUT_file_list (list): list of OUT files (containing orbital elements) to extract data from

    Returns:
        data (np.array): contains all orbital elements
        count (int): number of objects
    """
    data = []
    count = 0
    target_line1 = "ORBIT DETERMINATION WITHOUT PERTURBATIONS"
    target_line2 = "ORBIT DETERMINATION WITH PERTURBATIONS"
    
    num_obs = [] #number of observations 
    rms = [] #root mean square
    time_interval = [] # in seconds
    num_iter = [] #number of iterations
    P = [] 
    A = [] #semi major axis
    E = [] #eccentricity
    I = [] #inclination
    Node = [] #node
    Per = [] #perigee
    IPer = [] 
    
    
    for file in OUT_file_list: 
        with open(file, "r") as f: 
            f_data = f.read()
            data.append(f_data)
            
            # Reset file pointer to the beginning to read line by line
            f.seek(0) 
            lines = f.readlines()
            # Iterate through the lines to find the target line and process lines after it
            for i, line in enumerate(lines): 
                if line.strip() == target_line1.strip() or line.strip() == target_line2.strip():
                    count += 1
                    # Process the data after the target line
                    num_lines_to_extract = 12
                    extracted_data = lines[i+2:i+2+num_lines_to_extract]
                    num_obs.append(extracted_data[0])
                    rms.append(extracted_data[1])
                    time_interval.append(extracted_data[2])
                    num_iter.append(extracted_data[3])
                    P.append(extracted_data[5])
                    A.append(extracted_data[6])
                    E.append(extracted_data[7])
                    I.append(extracted_data[8])
                    Node.append(extracted_data[9])
                    Per.append(extracted_data[10])
                    IPer.append(extracted_data[11])
                        
    #remove all unnecessary errors and strings from the data            
    num_obs = np.array([int(line.split('=')[-1].strip()) for line in num_obs], dtype=int) 
    rms = np.array([float(line.split('=')[1].strip().replace('"', '')) for line in rms], dtype = float)
    time_interval = np.array([float(line.split('=')[1].strip().split()[0]) for line in time_interval], dtype = float)
    num_iter = np.array([int(line.split('=')[1].strip()) for line in num_iter], dtype = float)
    P = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in P], dtype = float)
    A = np.array([float(line.split('=')[1].strip().split()[0]) for line in A], dtype=float)   
    E = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in E], dtype=float)
    I = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in I], dtype=float)
    Node = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in Node], dtype=float)
    Per = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in Per], dtype=float)
    IPer = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in IPer], dtype=float)    
                    
    data = [num_obs, rms, time_interval, num_iter, P, A, E, I, Node, Per, IPer]
    data = np.array(data)
    return data, count

def get_orbele_and_date_from_celmech(OUT_file_list: list):
    """
    Open the OUT files from a list of files. Extracts orbital elements and stores them in a numpy array.
    Extracts dates and returns them as well.

    Args:
        OUT_file_list (list): List of OUT files (containing orbital elements) to extract data from.

    Returns:
        data (np.array): Contains all orbital elements.
        count (int): Number of objects.
        dates (list): Extracted observation dates converted to MJD.
    """
    data = []
    count = 0
    target_line = "ORBIT DETERMINATION WITHOUT PERTURBATIONS"

    num_obs = []  # Number of observations
    rms = []  # Root mean square
    time_interval = []  # In seconds
    num_iter = []  # Number of iterations
    P = []
    A = []  # Semi-major axis
    E = []  # Eccentricity
    I = []  # Inclination
    Node = []  # Node
    Per = []  # Perigee
    IPer = []

    dates = []  # To store extracted observation dates
    target_line_date = "OBSERVATIONS"
    stopping_line_data = "CIRCULAR ORBIT DETERMINATION WITH OBS NO"
    mask = []

    failed_counter = 0
    temp_dates_lengths = []  # List to store the lengths of temp_dates

    for file in OUT_file_list:
        with open(file, "r") as f:
            f_data = f.read()
            data.append(f_data)
            # Reset file pointer to the beginning to read line by line
            f.seek(0)
            lines = f.readlines()

            # Extract orbital elements
            skip_data = False  # Flag to skip current set of data
            temp_dates = []  # Temporary list to collect dates for the current block

            for line in lines:
                if line.strip() == target_line.strip():
                    mask.append(1)
                if target_line_date in line:
                    temp_dates = []  # Reset for new observation block
                    capturing = True
                    continue

                if stopping_line_data in line:
                    capturing = False  # Stop capturing dates
                    temp_dates_lengths.append(len(temp_dates))  # Store the length of temp_dates

                    if not skip_data:  # If no skip flag, commit temp_dates
                        dates.extend(temp_dates)
                    skip_data = False  # Reset skip flag
                    continue

                if capturing:
                    if ": ONLY 1 OBSERVATION FOR PASS" in line:
                        failed_counter += 1
                        mask.append(0)  # Mark as failure
                        skip_data = True  # Skip this block
                        temp_dates = []  # Discard any previously captured dates


                    parts = line.split()
                    if len(parts) >= 7:  # Ensure enough columns for date and time
                        date = " ".join(parts[3:9])  # Extract 'YYYY MM DD HH MM SS.SSS'
                        if date != 'YYYY MM DD HH MM SS.SSS':
                            temp_dates.append(date)

            # Extract other orbital data
            for i, line in enumerate(lines):
                if line.strip() == target_line.strip():
                    count += 1
                    # Process the data after the target line
                    num_lines_to_extract = 12
                    extracted_data = lines[i + 2:i + 2 + num_lines_to_extract]
                    num_obs.append(extracted_data[0])
                    rms.append(extracted_data[1])
                    time_interval.append(extracted_data[2])
                    num_iter.append(extracted_data[3])
                    P.append(extracted_data[5])
                    A.append(extracted_data[6])
                    E.append(extracted_data[7])
                    I.append(extracted_data[8])
                    Node.append(extracted_data[9])
                    Per.append(extracted_data[10])
                    IPer.append(extracted_data[11])

    # Remove all unnecessary errors and strings from the data
    num_obs = np.array([int(line.split('=')[-1].strip()) for line in num_obs], dtype=int)
    rms = np.array([float(line.split('=')[1].strip().replace('"', '')) for line in rms], dtype=float)
    time_interval = np.array([float(line.split('=')[1].strip().split()[0]) for line in time_interval], dtype=float)
    num_iter = np.array([int(line.split('=')[1].strip()) for line in num_iter], dtype=float)
    P = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in P], dtype=float)
    A = np.array([float(line.split('=')[1].strip().split()[0]) for line in A], dtype=float)
    E = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in E], dtype=float)
    I = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in I], dtype=float)
    Node = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in Node], dtype=float)
    Per = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in Per], dtype=float)
    IPer = np.array([float(line.split('=')[1].split(' +/-')[0].strip()) for line in IPer], dtype=float)

    data = [num_obs, rms, time_interval, num_iter, P, A, E, I, Node, Per, IPer]
    data = np.array(data)

    mjd_dates = []
    for date in dates:
        mjd = calculations.date_to_mjd_manual(date)
        mjd_dates.append(mjd)

    #print(f"Number of objects failed in Celmech: {failed_counter}")
    #print(f"Number of objects not failed in Celmech: {len(num_obs)}")
    #print(f"Total (failed and not failed) in Celmech: {failed_counter + len(num_obs)}")
    #print(f"Lengths of temp_dates: {len(temp_dates_lengths)}")
    #print(f"Total number of dates: {np.sum(temp_dates_lengths)}")
    return data, count, mjd_dates, mask

def array_extender_orbital_data(filename: str):
    """
    Function to load orbital data from plugin *.pro files into numpy arrays.
    Args:
        filename (str): The *.pro file containing orbital data.

    Returns:
        data (np.array): np.array of np.arrays containing all the orbital data.
    """
    #Initialize lists to store the data
    objX = []
    objY = []
    objZ = []
    objVx = []
    objVy = []
    objVz = []
    a_vals = []
    e_vals = []
    i_vals = []
    raan_vals = []
    omega_vals = []
    nu_vals = []
    
    with open(filename, "r") as inp:
        # Read the data file line by line
        data = inp.readlines()

    # Start reading data after the header (assumed to be in the first 2 lines)
    for line in data[2:]:  # Skipping the first two lines: header and separator line
        if line.strip():  # Ensure line is not empty
            parts = line.split()
            if len(parts) == 12:
                objX.append(float(parts[0]))
                objY.append(float(parts[1]))
                objZ.append(float(parts[2]))
                objVx.append(float(parts[3]))
                objVy.append(float(parts[4]))
                objVz.append(float(parts[5]))
                a_vals.append(float(parts[6]))
                e_vals.append(float(parts[7]))
                i_vals.append(float(parts[8]))
                raan_vals.append(float(parts[9]))
                omega_vals.append(float(parts[10]))
                nu_vals.append(float(parts[11]))

    # Convert lists to numpy arrays
    objX = np.array(objX)
    objY = np.array(objY)
    objZ = np.array(objZ)
    objVx = np.array(objVx)
    objVy = np.array(objVy)
    objVz = np.array(objVz)
    a_vals = np.array(a_vals)
    e_vals = np.array(e_vals)
    i_vals = np.array(i_vals)
    raan_vals = np.array(raan_vals)
    omega_vals = np.array(omega_vals)
    nu_vals = np.array(nu_vals)

    # Return all data as a single array of arrays
    data = [objX, objY, objZ, objVx, objVy, objVz, a_vals, e_vals, i_vals, raan_vals, omega_vals, nu_vals]
    return data

def get_data_for_plotting_from_OUT_file(year: str, orbit: str, err: bool, ell: bool, compare: bool = False):
    """find the right file for specified year, orbit, err and ell. Extracts the data from
    that file and returns the arrays that are used or the i- omega plots and for the semi major
    axis plot. 

    Args:
        year (str): year of the data
        orbit (str): orbit type (geo, gto or followup) of the data
        err (bool): whether the desired data set contains noise or not. Noise levels were specified as input parameters for Celmech.
        ell (bool): If True: elliptical orbit. If False: the data contains only circular orbits.
        compare (bool, optional): Should only be true, if ell is false. This is used if the circular orbit data
        should be compared to the elliptical orbit data (obtain it through separate call of this function). 
        As elliptical orbits can only be calculated when num_obs >= 3, all other objects are excluded in the
        corresponding circular orbits, so they can be compared (for example in i- omega plot). Defaults to False.

    Returns:
        orbit_data_dict[orbit]["A"] (np.array): semi major axis data
        orbit_data_dict[orbit]["I"](np.array): inclination data
        orbit_data_dict[orbit]["Node"] (np.array): raan data
    """
    # Dictionary for storing orbit data separately
    orbit_data_dict = {
        "geo": {"num_obs": [], "rms": [], "time_interval": [], "num_iter": [], 
                "P": [], "A": [], "E": [], "I": [], "Node": [], "Per": [], "IPer": []},
        "gto": {"num_obs": [], "rms": [], "time_interval": [], "num_iter": [], 
                "P": [], "A": [], "E": [], "I": [], "Node": [], "Per": [], "IPer": []},
        "fol": {"num_obs": [], "rms": [], "time_interval": [], "num_iter": [], 
                "P": [], "A": [], "E": [], "I": [], "Node": [], "Per": [], "IPer": []}
    }
    
    files = []
    file = get_celmech_OUT_files(year, orbit, err, ell)
    file = os.path.join("input_celmech", file)
    files.append(file)
    orbit_data, number_of_obj = get_orbele_from_celmech(files)
    
    # Append the data to the corresponding orbit type in the dictionary
    orbit_data_dict[orbit]["num_obs"].append(orbit_data[0])
    orbit_data_dict[orbit]["rms"].append(orbit_data[1])
    orbit_data_dict[orbit]["time_interval"].append(orbit_data[2])
    orbit_data_dict[orbit]["num_iter"].append(orbit_data[3])
    orbit_data_dict[orbit]["P"].append(orbit_data[4])
    orbit_data_dict[orbit]["A"].append(orbit_data[5])
    orbit_data_dict[orbit]["E"].append(orbit_data[6])
    orbit_data_dict[orbit]["I"].append(orbit_data[7])
    orbit_data_dict[orbit]["Node"].append(orbit_data[8])
    orbit_data_dict[orbit]["Per"].append(orbit_data[9])
    orbit_data_dict[orbit]["IPer"].append(orbit_data[10])

    # Filtering for inclination and apogee
    inc_data = np.array(orbit_data_dict[orbit]["I"])
    node_data = np.array(orbit_data_dict[orbit]["Node"])
    ecc_data = np.array(orbit_data_dict[orbit]["E"])
    a_data = np.array(orbit_data_dict[orbit]["A"])
    num_obs = np.array(orbit_data_dict[orbit]["num_obs"])


    #Filtering for number of observations, only if compare is true and ell is False
    if compare and not ell: 
        min_num_obs = 3
        mask_num_obs = num_obs >= min_num_obs
        orbit_data_dict[orbit]["I"] = inc_data[mask_num_obs]
        orbit_data_dict[orbit]["Node"] = node_data[mask_num_obs]
        orbit_data_dict[orbit]["E"] = ecc_data[mask_num_obs]
        orbit_data_dict[orbit]["A"] = a_data[mask_num_obs]    
        
    maxinc = 22
    apogee_threshold = 10000  # in meters (10 km)

    mask = inc_data <= maxinc
    orbit_data_dict[orbit]["I"] = inc_data[mask]
    orbit_data_dict[orbit]["Node"] = node_data[mask]
    orbit_data_dict[orbit]["E"] = ecc_data[mask]
    orbit_data_dict[orbit]["A"] = a_data[mask]

    assert len(orbit_data_dict[orbit]["I"]) == len(orbit_data_dict[orbit]["Node"]), "Filtering went wrong"   

    apogee = orbit_data_dict[orbit]["A"] * (1 + orbit_data_dict[orbit]["E"])
    mask_apogee = apogee >= apogee_threshold
    orbit_data_dict[orbit]["I"] = orbit_data_dict[orbit]["I"][mask_apogee]
    orbit_data_dict[orbit]["Node"] = orbit_data_dict[orbit]["Node"][mask_apogee]
    orbit_data_dict[orbit]["A"] = orbit_data_dict[orbit]["A"][mask_apogee]

    return orbit_data_dict[orbit]["A"], orbit_data_dict[orbit]["I"], orbit_data_dict[orbit]["Node"]

def data_for_one_year_one_seed_old(year: str, seed: str):
    """used to get all the simulation data for a specific year and seed, not separated/sorted

    Args:
        year (str): year of data
        seed (str): seed of data (1, 2, 3 or 4)

    Returns:
        data_crs (np.array): crossing data for that year
        data_det (np.array): detection data for that year
    """
    year2 = year[2:]

    if int(year) == 2023 and int(seed) == 1: 
        GEO_file_crs = f"stat_Master_{year2}_geo_s{seed}_10cm.crs"
        GTO_file_crs = f"stat_Master_{year2}_gto_s{seed}_10cm.crs"
        followup_file_crs = f"stat_Master_{year2}_fol_s{seed}_10cm.crs"
        GEO_file_crs = os.path.join("input", GEO_file_crs)
        GTO_file_crs = os.path.join("input", GTO_file_crs)
        followup_file_crs = os.path.join("input", followup_file_crs)
    else: 
        GEO_file_crs = f"stat_Master_{year2}_geo_s{seed}.crs"
        GTO_file_crs = f"stat_Master_{year2}_gto_s{seed}.crs"
        followup_file_crs = f"stat_Master_{year2}_fol_s{seed}.crs"
        GEO_file_crs = os.path.join("input", GEO_file_crs)
        GTO_file_crs = os.path.join("input", GTO_file_crs)
        followup_file_crs = os.path.join("input", followup_file_crs)

    data_GTO_crs = array_extender(GTO_file_crs)
    data_GEO_crs = array_extender(GEO_file_crs)
    data_followup_crs = array_extender(followup_file_crs)

    data_crs = np.hstack([data_GEO_crs, data_GTO_crs, data_followup_crs])

    if int(year) == 2023 and int(seed) == 1:
        GEO_file_det = f"stat_Master_{year2}_geo_s{seed}_10cm.det"
        GTO_file_det = f"stat_Master_{year2}_gto_s{seed}_10cm.det"
        followup_file_det = f"stat_Master_{year2}_fol_s{seed}_10cm.det"
        GEO_file_det = os.path.join("input", GEO_file_det)
        GTO_file_det = os.path.join("input", GTO_file_det)
        followup_file_det = os.path.join("input", followup_file_det)
    else: 
        GEO_file_det = f"stat_Master_{year2}_geo_s{seed}.det"
        GTO_file_det = f"stat_Master_{year2}_gto_s{seed}.det"
        followup_file_det = f"stat_Master_{year2}_fol_s{seed}.det"
        GEO_file_det = os.path.join("input", GEO_file_det)
        GTO_file_det = os.path.join("input", GTO_file_det)
        followup_file_det = os.path.join("input", followup_file_det)

    data_GTO_det = array_extender(GTO_file_det)
    data_GEO_det = array_extender(GEO_file_det)
    data_followup_det = array_extender(followup_file_det)
    
    data_det = np.hstack([data_GEO_det, data_GTO_det, data_followup_det])
    
    return data_crs, data_det

class PopulationType(Enum):
    NEWPOP_TH3 = "npth3"
    PROPAGATE = "prop"
    NEWPOP = "newpo"
    TH3 = "new2"
    TH25 = "new"
    NORMAL = ""
    NEWPOP_JAN = "janpo"
    NEWPOP_MARCH = "marpo"

def data_for_one_year_one_seed(year: str, seed: str, population_type: PopulationType):
    """used to get all the simulation data for a specific year and seed, not separated/sorted

    Args:
        year (str): year of data
        seed (str): seed of data (1, 2, 3 or 4)

    Returns:
        data_crs (np.array): crossing data for that year
        data_det (np.array): detection data for that year
    """
    year2 = year[2:]
    
    if int(year2) < (18 if population_type == PopulationType.NEWPOP_TH3 else 19):
        suffix = ""
    else:
        suffix = f"_{population_type.value}"

    GEO_file_crs = f"stat_Master_{year2}_geo_s{seed}{suffix}.crs"
    GTO_file_crs = f"stat_Master_{year2}_gto_s{seed}{suffix}.crs"
    followup_file_crs = f"stat_Master_{year2}_fol_s{seed}{suffix}.crs"
    GEO_file_crs = os.path.join("input", GEO_file_crs)
    GTO_file_crs = os.path.join("input", GTO_file_crs)
    followup_file_crs = os.path.join("input", followup_file_crs)

    data_GTO_crs = array_extender(GTO_file_crs)
    data_GEO_crs = array_extender(GEO_file_crs)
    data_followup_crs = array_extender(followup_file_crs)

    data_crs = np.hstack([data_GEO_crs, data_GTO_crs, data_followup_crs])

    GEO_file_det = f"stat_Master_{year2}_geo_s{seed}{suffix}.det"
    GTO_file_det = f"stat_Master_{year2}_gto_s{seed}{suffix}.det"
    followup_file_det = f"stat_Master_{year2}_fol_s{seed}{suffix}.det"
    GEO_file_det = os.path.join("input", GEO_file_det)
    GTO_file_det = os.path.join("input", GTO_file_det)
    followup_file_det = os.path.join("input", followup_file_det)

    data_GTO_det = array_extender(GTO_file_det)
    data_GEO_det = array_extender(GEO_file_det)
    data_followup_det = array_extender(followup_file_det)
    
    data_det = np.hstack([data_GEO_det, data_GTO_det, data_followup_det])
    
    return data_crs, data_det

def data_returner(year: str, seed: str, population_type: PopulationType, direct_filenames: list = None):
    """Used to get all simulation data for one specific year and seed, separated into different orbit types.

    Args:
        year (str): Year of the data
        seed (str): Seed of the data (1, 2, 3, or 4)
        population_type (PopulationType): Type of population
        direct_filenames (list, optional): List of six filenames [GEO_crs, GTO_crs, followup_crs, GEO_det, GTO_det, followup_det]

    Returns:
        (np.arrays): Crossing and detection data separated into different orbit types
    """
    if direct_filenames is None: 
        directory = "input"
        year2 = year[2:]
        
        if int(year2) < (18 if population_type == PopulationType.NEWPOP_TH3 else 19):
            suffix = ""
        if population_type == PopulationType.NORMAL:
            suffix = ""
        else:
            suffix = f"_{population_type.value}"
        
        if int(year2) < 18: 
            # Construct file paths for .crs files
            GEO_file_crs = os.path.join(directory, f"stat_Master_{year2}_geo_s{seed}.crs")
            GTO_file_crs = os.path.join(directory, f"stat_Master_{year2}_gto_s{seed}.crs")
            followup_file_crs = os.path.join(directory, f"stat_Master_{year2}_fol_s{seed}.crs")
            # Construct file paths for .det files
            GEO_file_det = os.path.join(directory, f"stat_Master_{year2}_geo_s{seed}.det")
            GTO_file_det = os.path.join(directory, f"stat_Master_{year2}_gto_s{seed}.det")
            followup_file_det = os.path.join(directory, f"stat_Master_{year2}_fol_s{seed}.det")
        else:  # case where the new pop files were used
            GEO_file_crs = os.path.join(directory, f"stat_Master_{year2}_geo_s{seed}{suffix}.crs")
            GTO_file_crs = os.path.join(directory, f"stat_Master_{year2}_gto_s{seed}{suffix}.crs")
            followup_file_crs = os.path.join(directory, f"stat_Master_{year2}_fol_s{seed}{suffix}.crs")
            GEO_file_det = os.path.join(directory, f"stat_Master_{year2}_geo_s{seed}{suffix}.det")
            GTO_file_det = os.path.join(directory, f"stat_Master_{year2}_gto_s{seed}{suffix}.det")
            followup_file_det = os.path.join(directory, f"stat_Master_{year2}_fol_s{seed}{suffix}.det")
    else:
        print("Direct filenames", direct_filenames)
        GEO_file_crs, GTO_file_crs, followup_file_crs, GEO_file_det, GTO_file_det, followup_file_det = direct_filenames

    # Load .crs data
    data_GEO_crs = array_extender(GEO_file_crs) if GEO_file_crs else None
    data_GTO_crs = array_extender(GTO_file_crs) if GTO_file_crs else None
    data_followup_crs = array_extender(followup_file_crs) if followup_file_crs else None

    # Load .det data
    data_GEO_det = array_extender(GEO_file_det) if GEO_file_det else None
    data_GTO_det = array_extender(GTO_file_det) if GTO_file_det else None
    data_followup_det = array_extender(followup_file_det) if followup_file_det else None

    # Return the data
    return data_GEO_crs, data_GTO_crs, data_followup_crs, data_GEO_det, data_GTO_det, data_followup_det


def data_four_years_one_seed(data_crs_all_seeds: list, data_det_all_seeds: list, years: str, dir: str, title: str, seeds: list, population_type: PopulationType):
    """loops through all seeds and processes the 4 year data packages, creates all kind of plots

    Args:
        data_crs_all_seeds (list): list of numpy arrays, one array per seed, crossing data
        data_det_all_seeds (list): list of numpy arrays, one array per seed, detection data
        years (str): list of years to process
        dir (str): directory, where to store the plots
    Returns:
        
    """
    import main_frag_and_rest
    print(years) #print years to terminal, so you know whether the program is progressing or not
    
    number_years = f"{years[0]}-{years[-1]}"
        
    for i, seed in enumerate(seeds): #loop over all seeds
        #call main function, extracts the data and does all kinds of plots
        inc_det_s, inc_crs_s = main_frag_and_rest.main_magnitude_cut(
            data_crs_all_seeds[i], 
            data_det_all_seeds[i], 
            number_years, 
            "all orbit types", 
            title,
            f"seed {seed}", 
            dir
        )
            
        #from here on: separate data and make i omega plot sorted for orbit types     
        data_GEO_det = []
        data_GTO_det = []
        data_followup_det = []
        
        for year in years: 
            data_GEO_crs_, data_GTO_crs_, data_followup_crs_, data_GEO_det_, data_GTO_det_, data_followup_det_ = data_returner(str(year), str(seed), population_type)
            data_GEO_det.append(data_GEO_det_)
            data_GTO_det.append(data_GTO_det_)
            data_followup_det.append(data_followup_det_)
        
        data_GEO_det = np.hstack(data_GEO_det)
        data_GTO_det = np.hstack(data_GTO_det)
        data_followup_det = np.hstack(data_followup_det)
                        
        max_inc = 22
        min_mag = 14.5
        
        #do sorting on GEO
        inc_GEO_det = data_GEO_det[9]
        nod_GEO_det = data_GEO_det[12]
        sem_major_GEO_det = data_GEO_det[8]
        ecc_GEO_det = data_GEO_det[10]  
        sources_GEO_det = data_GEO_det[3]
        mag_GEO_det = data_GEO_det[20]
        sorted = sortdata.sort_for_apogee(sem_major_GEO_det, ecc_GEO_det, inc_GEO_det, nod_GEO_det, sources_GEO_det, mag_GEO_det)
        inc_GEO_det = sorted[0]
        nod_GEO_det = sorted[1]
        sources_GEO_det = sorted[2]
        mag_GEO_det = sorted[3]
        sorted = sortdata.sort_for_inclination(inc_GEO_det, max_inc, nod_GEO_det, sources_GEO_det, mag_GEO_det)
        inc_GEO_det = [i for i in inc_GEO_det if i < max_inc]
        nod_GEO_det = sorted[0]
        sources_GEO_det = sorted[1]
        mag_GEO_det = sorted[2]

        #plotting the MLI objects seperately, as they have high A/M and might have orbit peturbations due to solar radiation pressure
        plotting.i_omega_MLI_separately(inc_GEO_det, nod_GEO_det, sources_GEO_det, years, dir)
    
        TLE_inc_det, frag_inc_det, rest_inc_det= sortdata.sort_for_sources(inc_GEO_det, sources_GEO_det)
        inc_GEO_det = np.hstack([frag_inc_det, rest_inc_det])
        TLE_nod_det, frag_nod_det, rest_nod_det= sortdata.sort_for_sources(nod_GEO_det, sources_GEO_det)
        nod_GEO_det = np.hstack([frag_nod_det, rest_nod_det])
        TLE_mag_det, frag_mag_det, rest_mag_det= sortdata.sort_for_sources(mag_GEO_det, sources_GEO_det)
        mag_GEO_det = np.hstack([frag_mag_det, rest_mag_det])

        sorted = sortdata.sort_for_magnitudes(mag_GEO_det, min_mag, inc_GEO_det, nod_GEO_det)
        inc_GEO_det = sorted[0]
        nod_GEO_det = sorted[1]

        #do sorting on GTO
        inc_GTO_det = data_GTO_det[9]
        nod_GTO_det = data_GTO_det[12]
        sem_major_GTO_det = data_GTO_det[8]
        ecc_GTO_det = data_GTO_det[10]  
        sources_GTO_det = data_GTO_det[3]
        mag_GTO_det = data_GTO_det[20]
        sorted = sortdata.sort_for_apogee(sem_major_GTO_det, ecc_GTO_det, inc_GTO_det, nod_GTO_det, sources_GTO_det, mag_GTO_det)
        inc_GTO_det = sorted[0]
        nod_GTO_det = sorted[1]
        sources_GTO_det = sorted[2]
        mag_GTO_det = sorted[3]
        sorted = sortdata.sort_for_inclination(inc_GTO_det, max_inc, nod_GTO_det, sources_GTO_det, mag_GTO_det)
        inc_GTO_det = [i for i in inc_GTO_det if i < max_inc]
        nod_GTO_det = sorted[0]
        sources_GTO_det = sorted[1]
        mag_GTO_det = sorted[2]
        
        #plotting the MLI objects seperately, as they have high A/M and might have orbit peturbations due to solar radiation pressure
        plotting.i_omega_MLI_separately(inc_GTO_det, nod_GTO_det, sources_GTO_det, years, dir)
        
        TLE_inc_det, frag_inc_det, rest_inc_det= sortdata.sort_for_sources(inc_GTO_det, sources_GTO_det)
        inc_GTO_det = np.hstack([frag_inc_det, rest_inc_det])
        TLE_nod_det, frag_nod_det, rest_nod_det= sortdata.sort_for_sources(nod_GTO_det, sources_GTO_det)
        nod_GTO_det = np.hstack([frag_nod_det, rest_nod_det])
        TLE_mag_det, frag_mag_det, rest_mag_det= sortdata.sort_for_sources(mag_GTO_det, sources_GTO_det)
        mag_GTO_det = np.hstack([frag_mag_det, rest_mag_det])

        sorted = sortdata.sort_for_magnitudes(mag_GTO_det, min_mag, inc_GTO_det, nod_GTO_det)
        inc_GTO_det = sorted[0]
        nod_GTO_det = sorted[1]

        #do sorting on followup
        inc_fol_det = data_followup_det[9]
        nod_fol_det = data_followup_det[12]
        sem_major_fol_det = data_followup_det[8]
        ecc_fol_det = data_followup_det[10]  
        sources_fol_det = data_followup_det[3]
        mag_fol_det = data_followup_det[20]
        sorted = sortdata.sort_for_apogee(sem_major_fol_det, ecc_fol_det, inc_fol_det, nod_fol_det, sources_fol_det, mag_fol_det)
        inc_fol_det = sorted[0]
        nod_fol_det = sorted[1]
        sources_fol_det = sorted[2]
        mag_fol_det = sorted[3]
        sorted = sortdata.sort_for_inclination(inc_fol_det, max_inc, nod_fol_det, sources_fol_det, mag_fol_det)
        inc_fol_det = [i for i in inc_fol_det if i < max_inc]
        nod_fol_det = sorted[0]
        sources_fol_det = sorted[1]
        mag_fol_det = sorted[2]
        
        #plotting the MLI objects seperately, as they have high A/M and might have orbit peturbations due to solar radiation pressure
        plotting.i_omega_MLI_separately(inc_fol_det, nod_fol_det, sources_fol_det, years, dir)
                
        TLE_inc_det, frag_inc_det, rest_inc_det= sortdata.sort_for_sources(inc_fol_det, sources_fol_det)
        inc_fol_det = np.hstack([frag_inc_det, rest_inc_det])
        TLE_nod_det, frag_nod_det, rest_nod_det= sortdata.sort_for_sources(nod_fol_det, sources_fol_det)
        nod_fol_det = np.hstack([frag_nod_det, rest_nod_det])
        TLE_mag_det, frag_mag_det, rest_mag_det= sortdata.sort_for_sources(mag_fol_det, sources_fol_det)
        mag_fol_det = np.hstack([frag_mag_det, rest_mag_det])
        sorted = sortdata.sort_for_magnitudes(mag_fol_det, min_mag, inc_fol_det, nod_fol_det)
        inc_fol_det = sorted[0]
        nod_fol_det = sorted[1]

        #function to create i omega plot for separated orbit types
        plotting.i_omega_all_orbits(np.array(nod_GEO_det), np.array(nod_GTO_det), np.array(nod_fol_det), np.array(inc_GEO_det), np.array(inc_GTO_det), np.array(inc_fol_det), f"Simulated detections {number_years} {seed}", years, dir)
        
    plotting.i_omega_all_orbits(np.array(nod_GEO_det), np.array(nod_GTO_det), np.array(nod_fol_det), np.array(inc_GEO_det), np.array(inc_GTO_det), np.array(inc_fol_det), f"Simulated detections {number_years}", years, dir)#no seed in title

    return len(nod_GEO_det), len(nod_GTO_det), len(nod_fol_det)

#end of enum

def find_monthly_files(folder: str, year: int, orbit_type: str, seed: int):
    """Find all .crs and .det files in 'folder' for a given year, orbit, and seed."""
    yy = f"{year % 100:02d}"  # Last two digits of year, zero-padded
    crs_files = []
    det_files = []

    for fname in os.listdir(folder):
        if fname.startswith(f"stat_Master_{yy}_{orbit_type}_s{seed}_"):
            parts = fname.split("_")
            # We expect parts[5] to be the month + extension, e.g. '02.det' or '02.crs'
            if len(parts) >= 6:
                month_part = parts[5]
                # Check if month_part has a two-digit month prefix
                if len(month_part) >= 5 and month_part[:2].isdigit():
                    if month_part.endswith(".crs"):
                        crs_files.append(os.path.join(folder, fname))
                    elif month_part.endswith(".det"):
                        det_files.append(os.path.join(folder, fname))

    return sorted(crs_files), sorted(det_files)

def find_monthly_files_from_ESA(folder: str, year: int, month: int):
    """Find all .crs and .det files in 'folder' for a given year and month."""
    crs_files = []
    det_files = []
    month_str = f"{month:02d}"

    prefix = f"ESA-SDT-{year}{month_str}01"

    for fname in os.listdir(folder):
        if fname.startswith(prefix):
            if fname.endswith(".crs"):
                crs_files.append(os.path.join(folder, fname))
            elif fname.endswith(".det"):
                det_files.append(os.path.join(folder, fname))

    return sorted(crs_files), sorted(det_files)

def data_monthly_one_seed_for_ESA(
    data_crs_all_seeds: list,
    data_det_all_seeds: list,
    years: list,
    dir: str,
    title: str,
    seeds: list,
    monthly_files_by_year_and_seed: dict
):
    number_years = f"{years}"

    import main_frag_and_rest

    for i, seed in enumerate(seeds):
        data_followup_det = []
        data_followup_crs = []

        for year in years:
            key = (year, seed)
            if key not in monthly_files_by_year_and_seed:
                continue

            crs_files = monthly_files_by_year_and_seed[key][0]
            det_files = monthly_files_by_year_and_seed[key][1]

            for i in range(len(crs_files)):
                followup_file_crs = crs_files[i]
                followup_file_det = det_files[i] if i < len(det_files) else None

                GEO_file_crs = None
                GTO_file_crs = None
                GEO_file_det = None
                GTO_file_det = None

                data_GEO_crs_, data_GTO_crs_, data_followup_crs_, \
                data_GEO_det_, data_GTO_det_, data_followup_det_ = data_returner(
                    year, seed, None,
                    [
                        GEO_file_crs,
                        GTO_file_crs,
                        followup_file_crs,
                        GEO_file_det,
                        GTO_file_det,
                        followup_file_det,
                    ],
                )

                print(np.array(data_followup_crs_).shape, np.array(data_followup_det_).shape)

                data_followup_crs.append(data_followup_crs_)
                data_followup_det.append(data_followup_det_)

        data_followup_det = np.hstack(data_followup_det)
        print(np.array(data_followup_det).shape)

        def process_and_plot_ESA(data_det, label):
            #this sorting was compared to data from André Horstman (should be the same method as he uses)
            size = data_det[1]
            inc = data_det[9]
            nod = data_det[12]
            print("inc, nod", len(inc), len(nod))
            sma = data_det[8]
            ecc = data_det[10]
            src = data_det[3]
            mag = data_det[20]

            """sorted = sortdata.sort_for_apogee(sma, ecc, inc, nod, src, mag, size)
            inc = sorted[0]
            nod = sorted[1]
            src = sorted[2]
            mag = sorted[3]
            size = sorted[4]"""

            min_size = 0
            sorted = sortdata.sort_for_sizes(size, min_size, nod, src, mag, inc)
            nod, src, mag, inc = sorted

            sorted = sortdata.sort_for_inclination(inc, 1000, nod, src, mag, inc)
            nod, src, mag, inc = sorted

            """tle_inc, frag_inc, rest_inc = sortdata.sort_for_sources(inc, src)
            inc = np.hstack([frag_inc, rest_inc, tle_inc])
            tle_nod, frag_nod, rest_nod = sortdata.sort_for_sources(nod, src)
            nod = np.hstack([frag_nod, rest_nod, tle_nod])
            tle_mag, frag_mag, rest_mag = sortdata.sort_for_sources(mag, src)
            mag = np.hstack([frag_mag, rest_mag, tle_mag])"""
            inc, nod = sortdata.sort_for_magnitudes(mag, 14, inc, nod, max_mag=19)

            return np.array(inc), np.array(nod)

        inc_fol, nod_fol = process_and_plot_ESA(data_followup_det, "Followup")

    plotting.i_omega_all_orbits(
        nod_fol, np.array([]), np.array([]),
        inc_fol, np.array([]), np.array([]),
        f"Simulated detections {number_years} from ESA", years, dir, label = "Number of detections in ESA *.det file"
    )

    return len(nod_fol), 0, 0

def data_monthly_one_seed(data_crs_all_seeds: list, data_det_all_seeds: list, years: list, dir: str, title: str, seeds: list, monthly_files_by_year_and_seed: dict):
    number_years = f"{years}"

    import main_frag_and_rest

    for i, seed in enumerate(seeds):
        """inc_det_s, inc_crs_s = main_frag_and_rest.main_magnitude_cut(
            data_crs_all_seeds[i],
            data_det_all_seeds[i],
            number_years,
            "all orbit types",
            title,
            f"seed {seed}",
            dir
        )"""

        data_GEO_det = []
        data_GTO_det = []
        data_followup_det = []
        data_GEO_crs = []
        data_GTO_crs = []
        data_followup_crs = []

        for year in years:
            for i in range(len(monthly_files_by_year_and_seed[(year, seed)][0])):  # loop over months
                GEO_file_crs       = monthly_files_by_year_and_seed[(year, seed)][0][i]
                GEO_file_det       = monthly_files_by_year_and_seed[(year, seed)][1][i]
                GTO_file_crs       = monthly_files_by_year_and_seed[(year, seed)][2][i]
                GTO_file_det       = monthly_files_by_year_and_seed[(year, seed)][3][i]
                followup_file_crs  = monthly_files_by_year_and_seed[(year, seed)][4][i]
                followup_file_det  = monthly_files_by_year_and_seed[(year, seed)][5][i]

                data_GEO_crs_, data_GTO_crs_, data_followup_crs_, data_GEO_det_, data_GTO_det_, data_followup_det_ = data_returner(
                    year, seed, None,
                    [GEO_file_crs, GTO_file_crs, followup_file_crs, GEO_file_det, GTO_file_det, followup_file_det]
                )

                data_GEO_crs.append(data_GEO_crs_)
                data_GTO_crs.append(data_GTO_crs_)
                data_followup_crs.append(data_followup_crs_)

                data_GEO_det.append(data_GEO_det_)
                data_GTO_det.append(data_GTO_det_)
                data_followup_det.append(data_followup_det_)

        data_GEO_det = np.hstack(data_GEO_det)
        data_GTO_det = np.hstack(data_GTO_det)
        data_followup_det = np.hstack(data_followup_det)

        def process_and_plot(data_det, label):
            size = data_det[1]
            inc = data_det[9]
            nod = data_det[12]
            sma = data_det[8]
            ecc = data_det[10]
            src = data_det[3]
            mag = data_det[20]

            sorted = sortdata.sort_for_apogee(sma, ecc, inc, nod, src, mag, size)
            inc = sorted[0]
            nod = sorted[1]
            src = sorted[2]
            mag = sorted[3]
            size = sorted[4]

            min_size = 0 # in meters
            sorted = sortdata.sort_for_sizes(size, min_size, nod, src, mag, inc)
            nod, src, mag, inc = sorted

            sorted = sortdata.sort_for_inclination(inc, 40, nod, src, mag)
            inc = [i for i in inc if i < 40]
            nod, src, mag = sorted

            _, frag_inc, rest_inc = sortdata.sort_for_sources(inc, src)
            inc = np.hstack([frag_inc, rest_inc])
            _, frag_nod, rest_nod = sortdata.sort_for_sources(nod, src)
            nod = np.hstack([frag_nod, rest_nod])
            _, frag_mag, rest_mag = sortdata.sort_for_sources(mag, src)
            mag = np.hstack([frag_mag, rest_mag])
            inc, nod = sortdata.sort_for_magnitudes(mag, 14, inc, nod, max_mag = 19)

            return np.array(inc), np.array(nod)

        inc_GEO, nod_GEO = process_and_plot(data_GEO_det, "GEO")
        inc_GTO, nod_GTO = process_and_plot(data_GTO_det, "GTO")
        inc_fol, nod_fol = process_and_plot(data_followup_det, "Followup")

        plotting.i_omega_all_orbits(nod_GEO, nod_GTO, nod_fol, inc_GEO, inc_GTO, inc_fol,
                                   f"Simulated detections {number_years} {seed}", years, dir)

    plotting.i_omega_all_orbits(nod_GEO, nod_GTO, nod_fol, inc_GEO, inc_GTO, inc_fol,
                                f"Simulated detections {number_years}", years, dir)

    return len(nod_GEO), len(nod_GTO), len(nod_fol)

def read_metadata_file(metadata_filepath):
    """
    Reads the metadata file and returns a dictionary mapping ID (int) -> COSPAR ID (str).
    """
    id_to_cospar = {}
    with open(metadata_filepath, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            obj_id = int(parts[0])
            cospar = parts[2]
            id_to_cospar[obj_id] = cospar
    return id_to_cospar

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def i_omega_colored_by_cospar(data_det, title, years, cospar_dict, out_dir):
    """
    Plots inclination vs RAAN colored by COSPAR ID, with a legend below the plot.
    Also plots points with no COSPAR match in gray.
    """
    # Extract fields
    size = data_det[1]
    inc = data_det[9]
    nod = data_det[12]
    sma = data_det[8]
    ecc = data_det[10]
    src = data_det[3]
    mag = data_det[20]
    object_id = data_det[0]

    # Apply sorting and filtering steps
    inc, nod, src, mag, size, object_id = sortdata.sort_for_apogee(sma, ecc, inc, nod, src, mag, size, object_id)
    nod, src, mag, inc, object_id = sortdata.sort_for_sizes(size, 0.1, nod, src, mag, inc, object_id)
    inc, nod, src, mag, object_id = sortdata.sort_for_inclination(inc, 40, inc, nod, src, mag, object_id)
    inc = np.array(inc)
    nod = np.array(nod)
    object_id = np.array(object_id)

    # Source-based stacking
    _, frag_inc, rest_inc = sortdata.sort_for_sources(inc, src)
    _, frag_nod, rest_nod = sortdata.sort_for_sources(nod, src)
    _, frag_mag, rest_mag = sortdata.sort_for_sources(mag, src)  
    _, frag_id, rest_id = sortdata.sort_for_sources(object_id, src)
    inc = np.hstack([frag_inc, rest_inc])
    nod = np.hstack([frag_nod, rest_nod])
    mag = np.hstack([frag_mag, rest_mag])
    object_id = np.hstack([frag_id, rest_id])

    # Magnitude filter
    inc, nod, object_id = sortdata.sort_for_magnitudes(mag, 14, inc, nod, object_id, max_mag=19)

    # Split COSPAR matches
    filtered_inc, filtered_nod, final_cospars = [], [], []
    no_match_inc, no_match_nod = [], []
    for i, oid in enumerate(object_id):
        prefix = str(int(oid))[:3]
        cos = cospar_dict.get(int(prefix))
        if cos:
            filtered_inc.append(inc[i]); filtered_nod.append(nod[i]); final_cospars.append(cos)
        else:
            no_match_inc.append(inc[i]); no_match_nod.append(nod[i])
    filtered_inc = np.array(filtered_inc)
    filtered_nod = np.array(filtered_nod)
    no_match_inc = np.array(no_match_inc)
    no_match_nod = np.array(no_match_nod)

    # Prepare colors
    unique_cospars = sorted(set(final_cospars))
    colors_list = plotting.get_200_distinct_colors()
    cospar_counts = Counter(final_cospars)
    sorted_cospars = [c for c,_ in cospar_counts.most_common()]
    cospar_to_color = {c: colors_list[i] for i,c in enumerate(sorted_cospars)}
    colors = [cospar_to_color[c] for c in final_cospars]

    # Normalize RAAN to [-180,180]
    nod = np.array(nod)
    nod = np.mod(nod + 180, 360) - 180
    filtered_nod = np.mod(filtered_nod + 180, 360) - 180
    no_match_nod = np.mod(no_match_nod + 180, 360) - 180

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(filtered_nod, filtered_inc, c=colors, s=20, label='_nolegend_')
    if no_match_inc.size:
        plt.scatter(no_match_nod, no_match_inc, c='gray', s=20, label='No COSPAR match')

    plt.xlabel("RAAN [$\Omega$] [deg]")
    plt.ylabel("Inclination [$i$] [deg]")
    plt.title(title)

    # Gridlines (major + minor)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()

    # Legend below
    handles = [plt.Line2D([0],[0], marker='o', color='w', label=c,
                          markerfacecolor=cospar_to_color[c], markersize=8)
               for c in sorted_cospars]
    if no_match_inc.size:
        handles.append(plt.Line2D([0],[0], marker='o', color='w', label='No COSPAR match',
                                  markerfacecolor='gray', markersize=8))
    plt.legend(handles=handles, title='COSPAR ID',
               loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=min(len(handles), 6), frameon=False)

    plt.ylim(0, 40)
    plt.xlim(-180, 180)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.path.join(out_dir, f"{title}.png"), dpi=300)
    plt.close()

def data_monthly_one_seed_with_id(data_crs_all_seeds: list, data_det_all_seeds: list, years: list, dir: str, title: str, seeds: list, monthly_files_by_year_and_seed: dict, metafile_dict: dict):
    number_years = f"{years}"

    import main_frag_and_rest

    for i, seed in enumerate(seeds):
        """inc_det_s, inc_crs_s = main_frag_and_rest.main_magnitude_cut(
            data_crs_all_seeds[i],
            data_det_all_seeds[i],
            number_years,
            "all orbit types",
            title,
            f"seed {seed}",
            dir
        )"""

        data_GEO_det = []
        data_GTO_det = []
        data_followup_det = []
        data_GEO_crs = []
        data_GTO_crs = []
        data_followup_crs = []

        for year in years:
            for i in range(len(monthly_files_by_year_and_seed[(year, seed)][0])):  # loop over months
                GEO_file_crs       = monthly_files_by_year_and_seed[(year, seed)][0][i]
                GEO_file_det       = monthly_files_by_year_and_seed[(year, seed)][1][i]
                GTO_file_crs       = monthly_files_by_year_and_seed[(year, seed)][2][i]
                GTO_file_det       = monthly_files_by_year_and_seed[(year, seed)][3][i]
                followup_file_crs  = monthly_files_by_year_and_seed[(year, seed)][4][i]
                followup_file_det  = monthly_files_by_year_and_seed[(year, seed)][5][i]

                data_GEO_crs_, data_GTO_crs_, data_followup_crs_, data_GEO_det_, data_GTO_det_, data_followup_det_ = data_returner(
                    year, seed, None,
                    [GEO_file_crs, GTO_file_crs, followup_file_crs, GEO_file_det, GTO_file_det, followup_file_det]
                )

                data_GEO_crs.append(data_GEO_crs_)
                data_GTO_crs.append(data_GTO_crs_)
                data_followup_crs.append(data_followup_crs_)

                data_GEO_det.append(data_GEO_det_)
                data_GTO_det.append(data_GTO_det_)
                data_followup_det.append(data_followup_det_)

        data_GEO_det = np.hstack(data_GEO_det)
        data_GTO_det = np.hstack(data_GTO_det)
        data_followup_det = np.hstack(data_followup_det)

        cospar = metafile_dict
        print("here", cospar)
        i_omega_colored_by_cospar(data_GEO_det, f"GEO Detections {number_years}", years, cospar, dir)
        i_omega_colored_by_cospar(data_GTO_det, f"GTO Detections {number_years}", years, cospar, dir)
        i_omega_colored_by_cospar(data_followup_det, f"Followup Detections {number_years}", years, cospar, dir)

        #plotting.i_omega_all_orbits(nod_GEO, nod_GTO, nod_fol, inc_GEO, inc_GTO, inc_fol,
                                   # f"Simulated detections {number_years} {seed}", years, dir)

    #plotting.i_omega_all_orbits(nod_GEO, nod_GTO, nod_fol, inc_GEO, inc_GTO, inc_fol,
                               # f"Simulated detections {number_years}", years, dir)

    return 

def read_DISCOS_file(filename: str): 
    """function to read the DISCOS file. DISCOS files contain clusters from the simulations from Andre Horstmann. 
    
    Args:
        filename (str): Path to the space debris data file.
    
    Returns:
        data (np.array): a numpy array of numpy arrays with all the data
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extract column names from the first line
    header = lines[0].strip().split()
    
    ID, NR, SW, INT_DESI, SATNO, NA, TY, CL, ABC, MASS = [], [], [], [], [], [], [], [], [], []
    DIAMTR, SMA, ECC, INC, RAAN, AOP, TANO, DC, RCSFAC, RPOP1, RPOP2 = [], [], [], [], [], [], [], [], [], [], []
    SEED, Breakup_Epoch, Launch_Epoch, Perigee_Alt, Apogee_Alt = [], [], [], [], []
    
    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 20 and parts[0].replace('.', '', 1).isdigit():
            ID.append(float(parts[0]))
            NR.append(float(parts[1]))
            SW.append(float(parts[2]))
            INT_DESI.append(parts[3])
            SATNO.append(float(parts[4]))
            NA.append(float(parts[5]))
            TY.append(float(parts[6]))
            CL.append(float(parts[7]))
            ABC.append(float(parts[8]))
            MASS.append(float(parts[9]))
            DIAMTR.append(float(parts[10]))
            SMA.append(float(parts[11]))
            ECC.append(float(parts[12]))
            INC.append(float(parts[13]))
            RAAN.append(float(parts[14]))
            AOP.append(float(parts[15]))
            TANO.append(float(parts[16]))
            DC.append(float(parts[17]))
            RCSFAC.append(float(parts[18]))
            RPOP1.append(float(parts[19]))
            RPOP2.append(float(parts[20]))
            SEED.append(float(parts[21]))
            
            breakup_epoch = " ".join(parts[22:24])  
            launch_epoch = " ".join(parts[24:26])  
            Breakup_Epoch.append(breakup_epoch)
            Launch_Epoch.append(launch_epoch)
            
            Perigee_Alt.append(float(parts[26]))
            Apogee_Alt.append(float(parts[27]))
    
    data = [
        np.array(ID), np.array(NR), np.array(SW), np.array(INT_DESI), np.array(SATNO), np.array(NA),
        np.array(TY), np.array(CL), np.array(ABC), np.array(MASS), np.array(DIAMTR), np.array(SMA), np.array(ECC),
        np.array(INC), np.array(RAAN), np.array(AOP), np.array(TANO), np.array(DC), np.array(RCSFAC),
        np.array(RPOP1), np.array(RPOP2), np.array(SEED), np.array(Breakup_Epoch), np.array(Launch_Epoch),
        np.array(Perigee_Alt), np.array(Apogee_Alt)
    ]
    
    return data