import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
#print("Files in %r: %s" % (cwd, files)) 

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

#helper function used for all the plots
def save_unique_plot(file_path: str, directory: str) -> str:
    """
    Helper function to ensure a unique filename in the given directory.
    If the filename already exists, append _1, _2, etc., to make it unique.
    Returns the new filename.
    """
    base_name = os.path.basename(file_path) 
    base_path, extension = os.path.splitext(base_name)  # Split into base name and extension, extension e.g. .png
    new_file_path = os.path.join(directory, base_name)
    
    count = 1
    while os.path.exists(new_file_path):  # Check if the file already exists
        new_file_path = os.path.join(directory, f"{base_path}_{count}{extension}") 
        count += 1 
    
    return new_file_path

#source analysis

def source_hist(sources: np.array, title: str, directory: str, year: str, orbit_type: str):
    """creates a histogram for the different debris sources contained in the *.det or *.crs file

    Args:
        sources (np.array): contains the different sources
        title (str): histogram title
        directory (str): place to store the plot
        year (str): year of the source data, used for plot title and plot name
        orbit_type (str): orbit type, used for plot title and plot name
    """
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(height),
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    bin_edges = np.arange(1, 8) - 0.5
    plt.clf()
    plt.figure(figsize=(10, 6), dpi = 450)
    plt.xlabel('Source')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    plt.xticks(range(1, 7))    
    n, bins, patches = plt.hist(sources, bins=bin_edges, edgecolor='black', color = "lightcoral", label="1 = Fragments, 2 = SRM Slag,  \n 3 = NaK droplets, 4 = TLEs, \n 5 = Westford Needles, 6 = Insulation")
    autolabel(patches)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    #plt.show()
    file_path = f"source_hist_{year}_{orbit_type}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    
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
    # Extract semi-major axis and eccentricity arrays
    semi_major = all_data[semi_major_index]
    ecc = all_data[ecc_index]
    semi_major = all_data[semi_major_index].astype(float)
    ecc = all_data[ecc_index].astype(float)
    if len(semi_major) != len(ecc):
        raise ValueError("The lengths of semi_major and ecc must be the same.")

    # Calculate apogee from semi-major axis and eccentricity
    apogee = [a * (1 + e) for a, e in zip(semi_major, ecc)]

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

def sort_for_magnitudes(mag: np.array, min_mag: float, *arrays): 
    sorted_arrays =[]
    for array in arrays: 
        if len(array) != len(mag): 
            raise ValueError("All input arrays must have the same length as mag!")
        sorted_array = [array[i] for i in range(len(mag)) if mag[i] >= min_mag]
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
    valid_indices = [i for i in range(len(mag)) if mag[i] >= min_mag]

    # Filter all_data to only include objects with valid magnitudes
    filtered_data = all_data[:, valid_indices]

    return filtered_data

#all kinds of plots
def magnitude_plot(mag_crs: np.array, year: int, title: str, orbit_type: str, directory: str, mag_det: np.array = None): 
    """magnitude frequency plot

    Args:
        mag_crs (np.array): magnitudes of the crossing objects
        year (int): year for the title
        title (str): title of the plot
        orbit_type (str): orbit type for the title
        directory (str): place to store the plot
        mag_det (np.array, optional): magnitudes of the detection objects. Defaults to None.
    """
    plt.clf()
    if mag_det is None: 
        zeros_crs = np.sum(1 for i in mag_crs if i == 0)
        #print(f"Number of Mag = 0 objects in crossings: {zeros_crs}")
        mag_crs = np.array([i for i in mag_crs if i > 7 and i < 25])    
        bin_edges = np.arange(7.5, 24.5, 0.5) + 0.25
        plt.figure(figsize=(10, 6), dpi = 450)
        n1, bins1, patches1 = plt.hist(mag_crs, bins = bin_edges, edgecolor='black', color = "skyblue")
        plt.xlabel('Apparent magnitude [mag]')
        plt.ylabel('Frequency')
        plt.xticks(range(7, 24))
        plt.title(title + f"         Count: {len(mag_crs)}")
        plt.grid(True)
        #plt.show()
        file_path = f"magnitudes_{year}_{orbit_type}.png"
        file_path = save_unique_plot(file_path, directory)
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        
    if mag_det is not None: 
        zeros_crs = np.sum(1 for i in mag_crs if i == 0)
        zeros_det = np.sum(1 for i in mag_det if i == 0)
        #print(f"Number of Mag = 0 objects in crossings: {zeros_crs}")
        #print(f"Number of Mag = 0 objects in detections: {zeros_det}")
        mag_crs = np.array([i for i in mag_crs if i > 7 and i < 25])    
        mag_det = np.array([i for i in mag_det if i > 7 and i < 25])    
        bin_edges = np.arange(7.5, 24.5, 0.5) + 0.25
        plt.figure(figsize=(10, 6), dpi = 450)
        n1, bins1, patches1 = plt.hist(mag_crs, bins = bin_edges, edgecolor='black', label="crossings", color= "palegreen")
        n2, bins2, patches2 = plt.hist(mag_det, bins = bin_edges, edgecolor='black', label="detections", color= "palegreen")
        plt.xlabel('Apparent magnitude [mag]')
        plt.ylabel('Frequency')
        plt.xticks(range(7, 24))
        plt.title(title)
        plt.grid(True)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        #plt.show()
        file_path = f"magnitudes_{year}_{orbit_type}.png"
        file_path = save_unique_plot(file_path, directory)
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

def diameter_plot(diameter_crs: np.array, year: int, title: str, directory: str, orbit_type: str, diameter_det: np.array = None):
    """diameter frequency plot

    Args:
        diameter_crs (np.array): diameters of the crossing objects
        year (int): year for the title
        title (str): orbit type for the title
        directory (str): place to store the plot
        orbit_type (str): _description_
        diameter_det (np.array, optional): diameters of the detection objects. Defaults to None.
    """
    plt.clf()
    bin_edges = np.logspace(np.log10(diameter_crs.min()), np.log10(diameter_crs.max()), num=20) #scale is logarithmic!!
    
    plt.figure(figsize=(10, 6), dpi = 450)

    if diameter_det is None:
        n1, bins1, patches1 = plt.hist(diameter_crs, bins=bin_edges, edgecolor='black', color="palegreen")
    else:
        n1, bins1, patches1 = plt.hist(diameter_crs, bins=bin_edges, edgecolor='black', label="crossing objects", color="palegreen")
        n2, bins2, patches2 = plt.hist(diameter_det, bins=bin_edges, edgecolor='black', label="detected objects", color="lightblue", alpha=0.7)

    plt.xlabel('Diameter [m]')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.xticks(bin_edges, labels=[f'{int(x)}' if x.is_integer() else f'{x:.3f}' for x in bin_edges], rotation=45)
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    if diameter_det is not None:
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    #plt.show()
    file_path = f"diameter_{year}_{orbit_type}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    
def correlation_magnitudes_sizes(mag_crs: np.array, mag_det: np.array, diameter_crs: np.array, diameter_det: np.array, year: int, title: str, orbit_type: str, directory: str): 
    """diagram to illustrate the correlation between magnitudes and sizes 

    Args:
        mag_crs (np.array): magnitudes of the crossing objects
        mag_det (np.array): magnitudes of the detected objects
        diameter_crs (np.array): diameters of the crossing objects
        diameter_det (np.array): diameters of the detected objects
        year (int): year, used for title and name of plot
        title (str): title of the plot
        orbit_type (str): orbit type, used for title and name of plot
        directory (str): place to store the plot
    """
    plt.clf()
    plt.figure(figsize=(10, 6), dpi = 450)
    plt.title(title)
    plt.scatter(diameter_crs, mag_crs, s = 5, label = "crossings")
    plt.scatter(diameter_det, mag_det, s = 5, label = "detected")
    plt.xlabel('Diameter [m]')
    plt.ylabel('Magnitude [mag]')
    plt.xscale('log')
    plt.grid(True)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    #plt.show()
    file_path = f"correlation_{year}_{orbit_type}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

def i_omega_all_orbits(nod_GEO_det: np.array, nod_GTO_det: np.array, nod_followup_det: np.array, inc_GEO_det: np.array, inc_GTO_det: np.array, inc_followup_det: np.array, title: str, year: str, directory: str):
    nod_det_converted_GEO = np.where(nod_GEO_det >= 180, nod_GEO_det - 360, nod_GEO_det)
    nod_det_converted_GEO = np.mod(np.array(nod_det_converted_GEO) + 180, 360) - 180
    nod_det_converted_GTO = np.where(np.array(nod_GTO_det) >= 180, np.array(nod_GTO_det) - 360, np.array(nod_GTO_det))
    nod_det_converted_GTO = np.mod(np.array(nod_det_converted_GTO) + 180, 360) - 180
    nod_det_converted_fol = np.where(np.array(nod_followup_det) >= 180, np.array(nod_followup_det) - 360, np.array(nod_followup_det))
    nod_det_converted_fol = np.mod(np.array(nod_det_converted_fol) + 180, 360) - 180
    
    plt.clf()
    
    plt.figure(figsize=(10, 6), dpi = 450)

    plt.title(title)
                        
    plt.scatter(nod_det_converted_GEO, inc_GEO_det, c = "b", s = 5, label = f"Number of detections GEO survey: {len(inc_GEO_det)}")
    plt.scatter(nod_det_converted_GTO, inc_GTO_det, c = "r", s = 5, label = f"Number of detections GTO survey: {len(inc_GTO_det)}")
    plt.scatter(nod_det_converted_fol, inc_followup_det, c = "g", s = 5, label = f"Number of detections Followup: {len(inc_followup_det)}")

    plt.xlabel("Right Ascension of Ascending Node $\Omega$ [°]")
    plt.ylabel("Inclination [°]")
    plt.yticks(range(0, 23, 2))
    plt.xticks(range(-180, 181, 60))
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    plt.grid(True)
    #plt.show()
    file_path = f"omega_i_{year}_detected_only_orbits_labeled.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    
def i_omega_separate(nod_det: np.array, inc_det: np.array, date: str, title: str, year: str, orbit_type: str, directory: str, nod_crs: np.array = None, inc_crs: np.array = None):
    """creates the inclination- RAAN plots for both crossings and detections

    Args:
        nod_crs (np.array): RAAN values of crossing objects
        inc_crs (np.array): inclinations of crossing objects
        date (str): date, used for plot name
        title (str): title of the plot
        year (str): year, used for plot name
        orbit_type (str): orbit type, used for plot name
        directory (str): place to store the plot
        nod_det (np.array, optional): RAAN values of the detection objects. Defaults to None.
        inc_det (np.array, optional): inclinations of the detection objects. Defaults to None.
    """
    
    nod_det_converted = np.where(np.array(nod_det) >= 180, np.array(nod_det) - 360, np.array(nod_det))
    nod_det_converted = np.mod(np.array(nod_det_converted) + 180, 360) - 180
    
    nod_crs_converted = None if nod_crs is None else np.where(nod_crs > 180, nod_crs - 360, nod_crs)
    nod_crs_converted = None if nod_crs is None else np.mod(np.array(nod_crs_converted) + 180, 360) - 180
    
    plt.clf()
    if nod_crs_converted is not None and inc_crs is not None: 
        plt.figure(figsize=(10, 6), dpi = 450)
        plt.title(title)
        plt.scatter(nod_crs_converted, inc_crs, c = "r", s = 5, label = f"Number of crossings: {len(nod_crs_converted)}")
        plt.xlabel("Right Ascension of Ascending Node $\Omega$ [°]")
        plt.ylabel("Inclination [°]")
        plt.yticks(range(0, 23, 2))
        plt.xticks(range(-180, 181, 60))
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        plt.grid(True)
        #plt.show()
        file_path = f"omega_i_{year}_{orbit_type}_crossings_only.png"
        file_path = save_unique_plot(file_path, directory)
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        
    plt.figure(figsize=(10, 6), dpi = 450)
    new_title = title[:10] + "detections" + title[20:]
    new_title = title
    plt.title(new_title)
    plt.scatter(nod_det_converted, inc_det, c = "b", s = 5, label = f"Number of detections: {len(nod_det_converted)}")
    plt.xlabel("Right Ascension of Ascending Node $\Omega$ [°]")
    plt.ylabel("Inclination [°]")
    plt.yticks(range(0, 23, 2))
    plt.xticks(range(-180, 181, 60))
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    plt.grid(True)
    #plt.show()
    file_path = f"omega_i_{year}_{orbit_type}_detected_only.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    
def i_omega_with_ratio(nod_crs: np.array, inc_crs: np.array, date: str, title: str, year: str, orbit_type: str, directory: str, nod_det: np.array = None, inc_det: np.array = None):
    """creates the inclination- RAAN plot and prints the ratio of detected vs. crossing objects

    Args:
        nod_crs (np.array): RAAN values of crossing objects
        inc_crs (np.array): inclinations of crossing objects
        date (str): date, used for plot name
        title (str): title of the plot
        year (str): year, used for plot name
        orbit_type (str): orbit type, used for plot name
        directory (str): place to store the plot
        nod_det (np.array, optional): RAAN values of the detection objects. Defaults to None.
        inc_det (np.array, optional): inclinations of the detection objects. Defaults to None.
    """

    nod_det_converted = np.where(np.array(nod_det) >= 180, np.array(nod_det) - 360, np.array(nod_det))
    nod_det_converted = np.mod(np.array(nod_det_converted) + 180, 360) - 180
    
    nod_crs_converted = None if nod_crs is None else np.where(nod_crs > 180, nod_crs - 360, nod_crs)
    nod_crs_converted = None if nod_crs is None else np.mod(np.array(nod_crs_converted) + 180, 360) - 180
    
    plt.clf()
    if nod_det_converted is not None and inc_det is not None: 
        plt.figure(figsize=(10, 6), dpi = 450)
        plt.title(title)
        plt.scatter(nod_crs_converted, inc_crs, c = "r", s = 5, label = f"Number of crossings: {len(nod_crs_converted)}")
        plt.scatter(nod_det_converted, inc_det, c = "b", s = 5, label = f"Number of detections: {len(nod_det_converted)}")
        plt.xlabel("Right Ascension of Ascending Node $\Omega$ [°]")
        plt.ylabel("Inclination [°]")
        plt.yticks(range(0, 23, 2))
        plt.xticks(range(-180, 181, 60))
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
        plt.grid(True)
        #plt.show()
        file_path = f"omega_i_{year}_{orbit_type}.png"
        file_path = save_unique_plot(file_path, directory)
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        
        ratio = len(nod_det)/len(nod_crs)
        print(f"The ratio of detected objects vs. crossing objects: {ratio:.3f}")
        
        #print(f"Number of detection events: {len(nod_det)}")
        #print(f"Number of crossing events: {len(nod_crs)}")
    
    else: 
        plt.figure(figsize=(10, 6), dpi = 450)
        plt.title(f"Crossing objects, date: {date}")
        plt.scatter(nod_crs_converted, inc_crs, c = "r", s = 5, label = f"Number of crossings: {len(nod_crs_converted)}")
        plt.scatter(nod_det_converted, inc_det, c = "b", s = 5, label = f"Number of detections: {len(nod_det_converted)}")
        plt.xlabel("Right Ascension of Ascending Node $\Omega$ [°]")
        plt.ylabel("Inclination [°]")
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
        plt.yticks(range(0, 23, 2))
        plt.xticks(range(-180, 181, 60))
        plt.grid(True)
        file_path = f"omega_i_{year}_{orbit_type}.png"
        file_path = save_unique_plot(file_path, directory)
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        
def i_omega_with_eccentricity(nod: np.array, inc: np.array, ecc: np.array, title: str, year: str, directory: str): 
    nod_converted = np.where(np.array(nod) >= 180, np.array(nod) - 360, np.array(nod))
    nod_converted = np.mod(np.array(nod_converted) + 180, 360) - 180
    ecc = np.clip(ecc, 0, 1)
    plt.figure(figsize=(10, 6), dpi = 450)
    plt.title(title)
    scatter = plt.scatter(nod_converted, inc, c=ecc, s=5, cmap='viridis', label=f"Number of detections: {len(nod)}")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Eccentricity')
    plt.xlabel("Right Ascension of Ascending Node $\Omega$ [°]")
    plt.ylabel("Inclination [°]")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    plt.yticks(range(0, 23, 2))
    plt.xticks(range(-180, 181, 60))
    plt.grid(True)
    file_path = f"omega_i_with_ecc_{year}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

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
        [f"GEO {year}", f"{mean[0]:.1f} $\pm$ {std[0]:.1f}", f"{mean[1]:.1f} $\pm$ {std[1]:.1f}", f"{mean[2]:.1f} $\pm$ {std[2]:.1f}", f"{mean[3]:.1f} $\pm$ {std[3]:.1f}"],
        [f"GTO {year}", f"{mean[4]:.1f} $\pm$ {std[4]:.1f}", f"{mean[5]:.3f} $\pm$ {std[5]:.1f}", f"{mean[6]:.1f} $\pm$ {std[6]:.1f}", f"{mean[7]:.1f} $\pm$ {std[7]:.1f}"],
        [f"followup {year}", f"{mean[8]:.1f} $\pm$ {std[8]:.1f}", f"{mean[9]:.1f} $\pm$ {std[9]:.1f}", f"{mean[10]:.1f} $\pm$ {std[10]:.1f}", f"{mean[11]:.1f} $\pm$ {std[11]:.1f}"]
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
    geo_crossing_objects = count_unique_objects_from_array(geo_crs)[0]
    geo_detected_objects = count_unique_objects_from_array(geo_det)[0]

    gto_crossing_events = len(gto_crs)
    gto_detection_events = len(gto_det)
    gto_crossing_objects = count_unique_objects_from_array(gto_crs)[0]
    gto_detected_objects = count_unique_objects_from_array(gto_det)[0]

    followup_crossing_events = len(fol_crs)
    followup_detection_events = len(fol_det)
    followup_crossing_objects = count_unique_objects_from_array(fol_crs)[0]
    followup_detected_objects = count_unique_objects_from_array(fol_det)[0]

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
        [f"GEO {year}", f"{mean[0]:.3f} $\pm$ {std[0]:.3f}", f"{mean[1]:.3f} $\pm$ {std[1]:.3f}"],
        [f"GTO {year}", f"{mean[2]:.3f} $\pm$ {std[2]:.3f}", f"{mean[3]:.3f} $\pm$ {std[3]:.3f}"],
        [f"followup {year}", f"{mean[4]:.3f} $\pm$ {std[4]:.3f}", f"{mean[5]:.3f} $\pm$ {std[5]:.3f}"]
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

#uniqueness
def count_unique_objects_from_id(*filenames: str):
    """looks at the id of the objects in the *.det or *.crs files and removes duplicates
    Args: 
        filenames (strs): filenames to search for unique objects 

    Returns:
        len(unique_sorted_lines) (int): length of uniques
    """
    all_lines = []
    for filename in filenames:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file if '#' not in line]
            all_lines.extend(lines)
    
    # Extract unique lines based on the first column (number)
    unique_lines_dict = {}
    for line in all_lines:
        key = line.split()[0]  # Extracting the first column
        if key not in unique_lines_dict:
            unique_lines_dict[key] = line
    
    # Sort the unique lines based on the first column (number)
    unique_sorted_lines = sorted(unique_lines_dict.values())
    
    for filename in filenames:
        new_filename = 'unique_' + filename
        with open(new_filename, 'w') as new_file:
            for line in unique_sorted_lines:
                new_file.write(line + '\n')
    
    return len(unique_sorted_lines)

def count_unique_objects_from_array(*arrays):
    """counts unique objects from objects stored in an array

    Args: 
        arrays (np.arrays): arrays to count unique objects in
    
    Returns:
        unique_counts (np.array): contains the number of unique objects for each array
    """
    unique_counts = []
    
    for array in arrays:
        unique_objects_set = set(array)
        unique_counts.append(len(unique_objects_set))
                
    return unique_counts

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
                
import numpy as np

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

import os
import numpy as np

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

def data_sorter(array: np.array, semi_major_index: int, ecc_index: int, inc_index: int, mag_index: int, source_index: int = None):
    array[semi_major_index] = [float(i) for i in array[semi_major_index]]
    array[ecc_index] = [float(i) for i in array[ecc_index]]
    array[mag_index] = [float(i) for i in array[mag_index]]
    array = sort_for_apogee_all_data(array, semi_major_index, ecc_index)
    array = sort_for_inclination_all_data(array, inc_index, max_inc = 22)
    array = sort_for_magnitudes_all_data(array, mag_index, min_mag = 14.5)
    if source_index is not None: 
        array[source_index] = [int(i) for i in array[source_index]]
        array_TLE, array_frag, array_rest = sort_for_sources_all_data(array, source_index)
        return array_TLE, array_frag, array_rest
    else: 
        return array
    
def data_returner(year: str, seed: str):
    """used to get all simulation data for one specific year and seed, separated into the different orbit types

    Args:
        year (str): year of the data
        seed (str): seed of the data (1,2,3 or 4)

    Returns:
        (np.arrays): crossing and detection data separated into different orbit types
    """
    year2 = year[2:]

    if int(year) == 2023 and int(seed) == 1: #the files for 2023 and seed 1 are named differently
        GEO_file_crs = f"stat_Master_{year2}_geo_s{seed}_10cm.crs"
        GTO_file_crs = f"stat_Master_{year2}_gto_s{seed}_10cm.crs"
        followup_file_crs = f"stat_Master_{year2}_fol_s{seed}_10cm.crs"
    else: 
        GEO_file_crs = f"stat_Master_{year2}_geo_s{seed}.crs"
        GTO_file_crs = f"stat_Master_{year2}_gto_s{seed}.crs"
        followup_file_crs = f"stat_Master_{year2}_fol_s{seed}.crs"

    data_GTO_crs = array_extender(GTO_file_crs)
    data_GEO_crs = array_extender(GEO_file_crs)
    data_followup_crs = array_extender(followup_file_crs)
    
    if int(year) == 2023 and int(seed) == 1:
        GEO_file_det = f"stat_Master_{year2}_geo_s{seed}_10cm.det"
        GTO_file_det = f"stat_Master_{year2}_gto_s{seed}_10cm.det"
        followup_file_det = f"stat_Master_{year2}_fol_s{seed}_10cm.det"
    else: 
        GEO_file_det = f"stat_Master_{year2}_geo_s{seed}.det"
        GTO_file_det = f"stat_Master_{year2}_gto_s{seed}.det"
        followup_file_det = f"stat_Master_{year2}_fol_s{seed}.det"

    data_GTO_det = array_extender(GTO_file_det)
    data_GEO_det = array_extender(GEO_file_det)
    data_followup_det = array_extender(followup_file_det)
    
    return data_GEO_crs, data_GTO_crs, data_followup_crs, data_GEO_det, data_GTO_det, data_followup_det