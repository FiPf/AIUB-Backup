import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

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
    plt.figure(figsize=(10, 6), dpi = 200)
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
        plt.figure(figsize=(10, 6), dpi = 200)
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
        plt.figure(figsize=(10, 6), dpi = 200)
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
    
    plt.figure(figsize=(10, 6), dpi = 200)

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
    plt.figure(figsize=(10, 6), dpi = 200)
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
    """creates i omega plots where objects from the three different orbit types are displayed using different colors

    Args:
        nod_GEO_det (np.array): omega data for objects in GEO
        nod_GTO_det (np.array): omega data for objects in GTO
        nod_followup_det (np.array): omega data for followup objects
        inc_GEO_det (np.array): inclination data for objects in GEO
        inc_GTO_det (np.array): inclination data for objects in GTO
        inc_followup_det (np.array): inclination data for followup objects
        title (str): title of the plot
        year (str): year(s) of the data
        directory (str): where to store the plot
    """

    #bring the omega [°] in the right format (between -180° to 180° instead of 0° to 360°)
    nod_det_converted_GEO = np.where(nod_GEO_det >= 180, nod_GEO_det - 360, nod_GEO_det)
    nod_det_converted_GEO = np.mod(np.array(nod_det_converted_GEO) + 180, 360) - 180
    nod_det_converted_GTO = np.where(np.array(nod_GTO_det) >= 180, np.array(nod_GTO_det) - 360, np.array(nod_GTO_det))
    nod_det_converted_GTO = np.mod(np.array(nod_det_converted_GTO) + 180, 360) - 180
    nod_det_converted_fol = np.where(np.array(nod_followup_det) >= 180, np.array(nod_followup_det) - 360, np.array(nod_followup_det))
    nod_det_converted_fol = np.mod(np.array(nod_det_converted_fol) + 180, 360) - 180
    
    plt.clf()
    
    plt.figure(figsize=(10, 6), dpi = 200)

    plt.title(title)
                        
    plt.scatter(nod_det_converted_GEO, inc_GEO_det, c = "b", s = 5, label = f"Number of detections GEO survey: {len(inc_GEO_det)}")
    plt.scatter(nod_det_converted_GTO, inc_GTO_det, c = "r", s = 5, label = f"Number of detections GTO survey: {len(inc_GTO_det)}")
    plt.scatter(nod_det_converted_fol, inc_followup_det, c = "g", s = 5, label = f"Number of detections Followup: {len(inc_followup_det)}")

    plt.xlabel("Right Ascension of Ascending Node $\\Omega$ [°]")
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
        plt.figure(figsize=(10, 6), dpi = 200)
        plt.title(title)
        plt.scatter(nod_crs_converted, inc_crs, c = "r", s = 5, label = f"Number of crossings: {len(nod_crs_converted)}")
        plt.xlabel("Right Ascension of Ascending Node $\\Omega$ [°]")
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
        
    plt.figure(figsize=(10, 6), dpi = 200)
    new_title = title[:10] + "detections" + title[20:]
    new_title = title
    plt.title(new_title)
    plt.scatter(nod_det_converted, inc_det, c = "b", s = 5, label = f"Number of detections: {len(nod_det_converted)}")
    plt.xlabel("Right Ascension of Ascending Node $\\Omega$ [°]")
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
        plt.figure(figsize=(10, 6), dpi = 200)
        plt.title(title)
        plt.scatter(nod_crs_converted, inc_crs, c = "r", s = 5, label = f"Number of crossings: {len(nod_crs_converted)}")
        plt.scatter(nod_det_converted, inc_det, c = "b", s = 5, label = f"Number of detections: {len(nod_det_converted)}")
        plt.xlabel("Right Ascension of Ascending Node $\\Omega$ [°]")
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
        plt.figure(figsize=(10, 6), dpi = 200)
        plt.title(f"Crossing objects, date: {date}")
        plt.scatter(nod_crs_converted, inc_crs, c = "r", s = 5, label = f"Number of crossings: {len(nod_crs_converted)}")
        plt.scatter(nod_det_converted, inc_det, c = "b", s = 5, label = f"Number of detections: {len(nod_det_converted)}")
        plt.xlabel("Right Ascension of Ascending Node $\\Omega$ [°]")
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
    """colors the i omega plot depending on the eccentricity of the object
    
    Args:
        nod (np.array): omega data of the objects
        inc (np.array): inclination data of the objects
        ecc (np.array): eccentricity data of the objects
        title (str): title of the plot
        year (str): year(s) of the data
        directory (str): where to store the plot
    """
    nod_converted = np.where(np.array(nod) >= 180, np.array(nod) - 360, np.array(nod))
    nod_converted = np.mod(np.array(nod_converted) + 180, 360) - 180
    ecc = np.clip(ecc, 0, 1)
    plt.figure(figsize=(10, 6), dpi = 200)
    plt.title(title)
    scatter = plt.scatter(nod_converted, inc, c=ecc, s=5, cmap='viridis', label=f"Number of detections: {len(nod)}")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Eccentricity')
    plt.xlabel("Right Ascension of Ascending Node $\\Omega$ [°]")
    plt.ylabel("Inclination [°]")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    plt.yticks(range(0, 23, 2))
    plt.xticks(range(-180, 181, 60))
    plt.grid(True)
    file_path = f"omega_i_with_ecc_{year}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    
def i_omega_with_sem_maj(nod: np.array, inc: np.array, semmaj: np.array, title: str, year: str, directory: str):
    """colors the i omega plot depending on the semi major axis of the objects

    Args:
        nod (np.array): omega data of the objects
        inc (np.array): inclination data of the objects
        semmaj (np.array): semi major axis data of the objects
        title (str): title of the plot
        year (str): year(s) of the data
        directory (str): where to store the plot
    """
    
    nod_converted = np.where(np.array(nod) >= 180, np.array(nod) - 360, np.array(nod))
    nod_converted = np.mod(np.array(nod_converted) + 180, 360) - 180
    plt.figure(figsize=(10, 6), dpi = 200)
    plt.title(title)
    scatter = plt.scatter(nod_converted, inc, c=semmaj, s=5, cmap='hot', label=f"Number of detections: {len(nod)}")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Semi-major axis [km]')
    plt.xlabel("Right Ascension of Ascending Node $\\Omega$ [°]")
    plt.ylabel("Inclination [°]")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    plt.yticks(range(0, 23, 2))
    plt.xticks(range(-180, 181, 60))
    plt.grid(True)
    file_path = f"omega_i_with_semmaj_{year}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    
def i_omega_with_magnitudes(nod: np.array, inc: np.array, magnitudes: np.array, title: str, year: str, directory: str):
    """colors the i omega plot according to the magnitude of the objects

    Args:
        nod (np.array): omega data of the objects
        inc (np.array): inclination data of the objects
        magnitudes (np.array): magnitude data of the objects
        title (str): title of the plot
        year (str): year(s) of the data
        directory (str): where to store the plot
    """
    nod_converted = np.where(np.array(nod) >= 180, np.array(nod) - 360, np.array(nod))
    nod_converted = np.mod(np.array(nod_converted) + 180, 360) - 180
    plt.figure(figsize=(10, 6), dpi = 200)
    plt.title(title)
    scatter = plt.scatter(nod_converted, inc, c=magnitudes, s=5, cmap='plasma', label=f"Number of detections: {len(nod)}")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Apparent magnitude [mag]')
    plt.xlabel("Right Ascension of Ascending Node $\\Omega$ [°]")
    plt.ylabel("Inclination [°]")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    plt.yticks(range(0, 23, 2))
    plt.xticks(range(-180, 181, 60))
    plt.grid(True)
    file_path = f"omega_i_with_mag_{year}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    
def i_omega_joined(first_nod: np.array, second_nod: np.array, first_inc: np.array, second_inc: np.array, title: str, year: str, first_label: str, second_label: str, directory: str):
    """used to plot two different i omega data sets in one plot
    example: simulations using orbital elements from celmech (circular) vs. simulations using elliptical elements

    Args:
        first_nod (np.array): omega data of first data set
        second_nod (np.array): omega data of second data set
        first_inc (np.array): inc data of first data set
        second_inc (np.array): inc data of second data set
        title (str): title for the plot
        year (str): year of the data
        first_label (str): label for scatter plot for first data set
        second_label (str): label for scatter plot for second data set
        directory (str): directory where to store the plot
    """
    
    first_nod_converted = np.where(np.array(first_nod) >= 180, np.array(first_nod) - 360, np.array(first_nod))
    first_nod_converted = np.mod(np.array(first_nod_converted) + 180, 360) - 180
    
    second_nod_converted = np.where(np.array(second_nod) >= 180, np.array(second_nod) - 360, np.array(second_nod))
    second_nod_converted = np.mod(np.array(second_nod_converted) + 180, 360) - 180
    
    plt.figure(figsize=(10, 6), dpi = 200)
    plt.title(title)
    plt.scatter(first_nod_converted, first_inc, c="olive", s=5, label=f"Number of detections {first_label}: {len(first_nod)}")
    plt.scatter(second_nod_converted, second_inc, c="deeppink", s=5, label=f"Number of detections {second_label}: {len(second_nod)}")
    plt.xlabel("Right Ascension of Ascending Node $\\Omega$ [°]")
    plt.ylabel("Inclination [°]")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    plt.yticks(range(0, 23, 2))
    plt.xticks(range(-180, 181, 60))
    plt.grid(True)
    file_path = f"omega_i_joined_{year}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    
def compare_semi_major_plot(first_a: np.array, second_a: np.array, title: str, year: str, orbit_type: str, first_label: str, second_label: str, directory: str): 
    """plot two datasets of semi major axis (same size) into one graph to compare them 

    Args:
        first_a (np.array): semi major axis of first dataset
        second_a (np.array): semi major axis of second dataset
        title (str): title of the plot, count of object is added
        year (str): year of the data
        orbit_type (str): orbit type of the data
        first_label (str): label of first dataset
        second_label (str): label of second dataset
    """
    plt.figure(figsize=(10, 6), dpi = 200)
    plt.title(title)
    title = title + f"Count: {len(first_a)}"
    assert len(first_a) == len(second_a), "lengths of semi major axis array must match up!"
    x_vals = np.linspace(0, len(first_a), len(first_a))
    plt.scatter(x_vals, first_a, c="olive", s=5, label=f"Semi major axis {first_label}")
    plt.scatter(x_vals, second_a, c="deeppink", s=5, label=f"Semi major axis {second_label}")
    plt.xlabel("Element index")
    plt.ylabel("Semi major axis")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    plt.grid(True)
    file_path = f"a_comparison_{year}_{orbit_type}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def i_omega_joined_with_eccentricity(first_nod: np.array, second_nod: np.array, first_inc: np.array, second_inc: np.array, first_ecc: np.array, second_ecc: np.array, title: str, year: str, first_label: str, second_label: str, directory: str):
    """Used to plot two different i-omega data sets in one plot with a single eccentricity color bar, and warnings for invalid eccentricity values.
    
    Args:
        first_nod (np.array): omega data of first data set
        second_nod (np.array): omega data of second data set
        first_inc (np.array): inclination data of first data set
        second_inc (np.array): inclination data of second data set
        first_ecc (np.array): eccentricity data of first data set
        second_ecc (np.array): eccentricity data of second data set
        title (str): title for the plot
        year (str): year of the data
        first_label (str): label for scatter plot for first data set
        second_label (str): label for scatter plot for second data set
        directory (str): directory where to store the plot
    """
    
    # Check and warn if eccentricity values are out of range [0, 1]
    def check_eccentricity_range(eccentricity_array, label):
        for ecc in eccentricity_array: 
            if ecc < 0 or ecc > 1: 
                print("WARNING. Ecc out of bound!")
                print(eccentricity_array)
                break
        
    # Check for both datasets
    check_eccentricity_range(first_ecc, "First Data Set")
    check_eccentricity_range(second_ecc, "Second Data Set")
    
    # Convert nodes
    first_nod_converted = np.where(np.array(first_nod) >= 180, np.array(first_nod) - 360, np.array(first_nod))
    first_nod_converted = np.mod(np.array(first_nod_converted) + 180, 360) - 180
    
    second_nod_converted = np.where(np.array(second_nod) >= 180, np.array(second_nod) - 360, np.array(second_nod))
    second_nod_converted = np.mod(np.array(second_nod_converted) + 180, 360) - 180
    
    # Clip eccentricity to valid range [0, 1]
    first_ecc = np.clip(first_ecc, 0, 1)
    second_ecc = np.clip(second_ecc, 0, 1)

    # Combine the eccentricity data from both datasets
    combined_nod = np.concatenate([first_nod_converted, second_nod_converted])
    combined_inc = np.concatenate([first_inc, second_inc])
    combined_ecc = np.concatenate([first_ecc, second_ecc])

    # Plotting
    plt.figure(figsize=(10, 6), dpi = 200)
    plt.title(title)
    
    # Scatter plot for both datasets (colored by combined eccentricity)
    scatter = plt.scatter(combined_nod, combined_inc, c=combined_ecc, s=5, cmap='viridis', label=f"Number of detections {first_label}: {len(first_nod)} + {second_label}: {len(second_nod)}")
    
    # Colorbar for the combined eccentricity
    cbar = plt.colorbar(scatter)
    cbar.set_label('Eccentricity (Both Data Sets)')

    # Labels and grid
    plt.xlabel("Right Ascension of Ascending Node $\\Omega$ [°]")
    plt.ylabel("Inclination [°]")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    plt.yticks(range(0, 23, 2))
    plt.xticks(range(-180, 181, 60))
    plt.grid(True)
    
    # Save the plot
    file_path = f"omega_i_joined_ecc_{year}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
