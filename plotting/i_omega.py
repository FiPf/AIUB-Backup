import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
from plothelp import save_unique_plot

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
    plt.scatter(first_nod_converted, first_inc, c="b", s=5, label=f"Number of detections {first_label}: {len(first_nod)}")
    plt.scatter(second_nod_converted, second_inc, c="r", s=5, label=f"Number of detections {second_label}: {len(second_nod)}")
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

def i_omega_MLI_separately(inc: np.array, raan: np.array, sources: np.array, year: str, directory: str):
    """
    Plots RAAN vs. Inclination, separating MLI objects for visual distinction.

    Args:
        inc (np.array): Inclination values of the objects.
        raan (np.array): RAAN values of the objects.
        sources (np.array): Array indicating MLI status (e.g., 6 for MLI).
        year (str): Year for labeling the plot.
        directory (str): Directory to save the plot.
    """
    # Convert RAAN to the range [-180, 180]
    
    inc = np.array(inc)
    raan = np.array(raan)
    sources = np.array(sources)
    
    #Remove the TLEs!
    valid_mask = sources != 4
    inc = inc[valid_mask]
    raan = raan[valid_mask]
    sources = sources[valid_mask]
    
    raan_converted = np.mod(np.where(raan >= 180, raan - 360, raan) + 180, 360) - 180

    # Filtering based on sources (assuming 6 indicates MLI)
    mli_mask = sources == 6
    inc_MLI = inc[mli_mask]
    raan_MLI = raan_converted[mli_mask]

    inc_rest = inc[~mli_mask]
    raan_rest = raan_converted[~mli_mask]

    # Plotting
    plt.figure(figsize=(10, 6), dpi=200)
    plt.title(f"RAAN vs Inclination (MLI Highlighted) - {year}")
    plt.scatter(raan_rest, inc_rest, c="b", s=5, label="Non-MLI")
    plt.scatter(raan_MLI, inc_MLI, c="r", s=5, label="MLI")

    plt.xlabel("RAAN [°]")
    plt.ylabel("Inclination [°]")
    plt.ylim(0, 22)
    plt.xlim(-180, 180)
    plt.grid(True)
    plt.legend()

    file_path = f"MLI_separate_{year}.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    
def i_omega_per_size(inc: np.array, raan: np.array, diameter: np.array, year: str, directory: str, separate: bool):
    """makes the i_omega plot, separating the objects based on their diameter/size. 

    Args:
        inc (np.array): inclination data
        raan (np.array): node data
        diameter (np.array): diameter data
        year (str): year of the data
        directory (str): where to store the plots
        separate (bool): If True, every size range has its own plot. If False, all sizes end up 
        in the same plot with different colors.
    """
    inc = np.array(inc)
    raan = np.array(raan)
    diameter = np.array(diameter)
    
    raan_converted = np.mod(np.where(raan >= 180, raan - 360, raan) + 180, 360) - 180

    size_ranges_lower_bounds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    # If separate is true, create individual plots for each size range
    if separate:
        for lower_bound in size_ranges_lower_bounds:
            upper_bound = size_ranges_lower_bounds[size_ranges_lower_bounds.index(lower_bound) + 1] if size_ranges_lower_bounds.index(lower_bound) + 1 < len(size_ranges_lower_bounds) else float('inf')
            
            mask = (diameter >= lower_bound) & (diameter < upper_bound)
            inc_filtered = inc[mask]
            raan_filtered = raan_converted[mask]
            
            # Skip if no data in the range
            if len(inc_filtered) == 0:
                continue

            plt.figure(figsize=(10, 6), dpi=200)
            plt.title(f"RAAN vs Inclination (Size: {lower_bound} - {upper_bound} m) - {year}")
            plt.scatter(raan_filtered, inc_filtered, c="b", s=5)

            plt.xlabel("RAAN [°]")
            plt.ylabel("Inclination [°]")
            plt.ylim(0, 22)
            plt.xlim(-180, 180)
            plt.grid(True)

            file_path = f"RAAN_vs_Inc_size_{lower_bound}_to_{upper_bound}_{year}.png"
            file_path = save_unique_plot(file_path, directory)
            plt.savefig(file_path, bbox_inches="tight")
            plt.close()

    # If separate is false, combine all sizes in one plot with different colors
    else:
        plt.figure(figsize=(10, 6), dpi=200)
        plt.title(f"RAAN vs Inclination (All Sizes) - {year}")

        # Plot each size range with a different color
        for lower_bound in size_ranges_lower_bounds:
            upper_bound = size_ranges_lower_bounds[size_ranges_lower_bounds.index(lower_bound) + 1] if size_ranges_lower_bounds.index(lower_bound) + 1 < len(size_ranges_lower_bounds) else float('inf')
            
            mask = (diameter >= lower_bound) & (diameter < upper_bound)
            inc_filtered = inc[mask]
            raan_filtered = raan_converted[mask]

            if len(inc_filtered) == 0:
                continue

            plt.scatter(raan_filtered, inc_filtered, s=5, label=f"{lower_bound} - {upper_bound} m")

        plt.xlabel("RAAN [°]")
        plt.ylabel("Inclination [°]")
        plt.ylim(0, 22)
        plt.xlim(-180, 180)
        plt.grid(True)
        plt.legend()

        file_path = f"RAAN_vs_Inc_all_sizes_{year}.png"
        file_path = save_unique_plot(file_path, directory)
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()