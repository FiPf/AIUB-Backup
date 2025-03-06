import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
from plot_help import save_unique_plot

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

def plot_DISCOS_file(raan: np.array, inc: np.array, ecc: np.array, breakup_epoch: np.array, title: str, directory: str): 
    """DISCOS files were provided by Andre Horstmann, they contain the breakup events (orbital parameters and epochs) used
    for the generation of the *.pop files. We want to compare those with our observations/simulations. 

    Args:
        raan (np.array): array of raan values
        inc (np.array): array of inc values
        ecc (np.array): array of ecc values
        breakup_epoch (np.array): breakup epoch 
        title (str): title for the plot
        directory (str): directory where to store the plot
    """
    raan_converted = np.where(np.array(raan) >= 180, np.array(raan) - 360, np.array(raan))
    raan_converted = np.mod(np.array(raan_converted) + 180, 360) - 180
    
    breakup_years = [ele[:4] for ele in breakup_epoch]  # Extract year from the timestamp
    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.title(title)
    scatter = plt.scatter(raan_converted, inc, c=ecc, cmap="viridis", s=30, vmin = 0, vmax = 1)
    plt.colorbar(label="Eccentricity")
    
    plt.xlabel("RAAN [°]")
    plt.ylabel("Inclination [°]")
    plt.ylim(0, 22)
    plt.xlim(-180, 180)
    plt.grid(True)
    
    # Add text labels for breakup years
    for x, y, year in zip(raan_converted, inc, breakup_years):
        plt.text(x, y, year, fontsize=8, ha="right", va="bottom", color="black", alpha=0.7)

    file_path = "population_clusters.png"
    file_path = save_unique_plot(file_path, directory)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()