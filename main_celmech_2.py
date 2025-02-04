import numpy as np
import unique_objects
import sortdata
import getdata 
import plotting
import calculations
import os
from getdata import PopulationType, data_returner

def main_celmech_2(year:str, dir: str, err:bool, ell: bool): 
    """Function that solves the Kreisbahnproblem. Unpacks Celmech and crs data, sorts them to match in object numbers, 
    creates plots and prints numbers. 

    Args:
        year (str): year of the data
        dir (str): where to store the plot
        err (bool): whether to artificially include errors in the Celmech caluclations, should always be set to False!
        ell (bool): If True, Celmech calculated elliptical orbits, if False, Celmech calculated circular orbits. 
    """    
    population_type = PopulationType.NORMAL
    orbit_type_list = ["geo", "gto", "fol"]
    total_num_of_objects = 0
    yy = year[2:]
    
    #Unpack Celmech output data
    # Dictionary for storing orbit data separated by orbit type (Celmech output data)
    orbit_data_dict = {
        "geo": {"num_obs": [], "rms": [], "time_interval": [], "num_iter": [], 
                "P": [], "A": [], "E": [], "I": [], "Node": [], "Per": [], "IPer": []},
        "gto": {"num_obs": [], "rms": [], "time_interval": [], "num_iter": [], 
                "P": [], "A": [], "E": [], "I": [], "Node": [], "Per": [], "IPer": []},
        "fol": {"num_obs": [], "rms": [], "time_interval": [], "num_iter": [], 
                "P": [], "A": [], "E": [], "I": [], "Node": [], "Per": [], "IPer": []}
    }

    title_appendix = " with error" if err else " without error"
    failed_masks = []
    
    # Loop to gather data for each orbit type from the Celmech Output
    for orbit in orbit_type_list:
        files = []
        file = getdata.get_celmech_OUT_files(year, orbit, err, ell) #find the right Celmech file for given year and orbit type
        file = os.path.join("input_celmech", file)
        files.append(file)
        orbit_data, number_of_obj, dates, failed_mask = getdata.get_orbele_and_date_from_celmech(files) #extract orbital elements from Celmech files
        failed_masks.append(failed_mask) #failed mask: mask for how many orbit determinations failed in Celmech
        
        total_num_of_objects += number_of_obj
        
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
        
        maxinc = 22 #cut off all data at greater inclinations
        apogee_threshold = 10000  # in kilometers (10k km)

        # Filtering for inclination and apogee
        inc_data = np.array(orbit_data_dict[orbit]["I"])
        node_data = np.array(orbit_data_dict[orbit]["Node"])
        ecc_data = np.array(orbit_data_dict[orbit]["E"])
        a_data = np.array(orbit_data_dict[orbit]["A"])
    
        mask = inc_data <= maxinc
        orbit_data_dict[orbit]["I"] = inc_data[mask]
        orbit_data_dict[orbit]["Node"] = node_data[mask]
        orbit_data_dict[orbit]["E"] = ecc_data[mask]
        orbit_data_dict[orbit]["A"] = a_data[mask]
        """
        assert len(orbit_data_dict[orbit]["I"]) == len(orbit_data_dict[orbit]["Node"]), "Filtering went wrong"   

        apogee = orbit_data_dict[orbit]["A"] * (1 + orbit_data_dict[orbit]["E"])
        mask_apogee = apogee >= apogee_threshold
        orbit_data_dict[orbit]["I"] = orbit_data_dict[orbit]["I"][mask_apogee]
        orbit_data_dict[orbit]["Node"] = orbit_data_dict[orbit]["Node"][mask_apogee]"""

    geo_crs, gto_crs, fol_crs, geo_det, gto_det, fol_det = data_returner(year, 4, population_type)

    geo_crs = np.array(geo_crs)
    geo_crs = sortdata.remove_zero_background_mag(geo_crs, background_mag_index = 21, mag_index = 20, illumination_index = 19)
    geo_crs = sortdata.sort_for_inclination_all_data(geo_crs, 9, 22)
    geo_crs, orbit_data_dict['geo']['I'], orbit_data_dict['geo']['Node'] = sortdata.inclination_filter(
        geo_crs, 
        np.column_stack((orbit_data_dict['geo']['A'], orbit_data_dict['geo']['E'], 
                        orbit_data_dict['geo']['I'], orbit_data_dict['geo']['Node'])), 
        failed_masks[0], 
        maxinc
    )
    geo_inc = geo_crs[9]
    geo_nod = geo_crs[12]
    
    gto_crs = np.array(gto_crs)
    gto_crs = sortdata.remove_zero_background_mag(gto_crs, background_mag_index = 21, mag_index = 20, illumination_index = 19)
    gto_crs = sortdata.sort_for_inclination_all_data(gto_crs, 9, 22)
    gto_crs, orbit_data_dict['gto']['I'], orbit_data_dict['gto']['Node'] = sortdata.inclination_filter(
        gto_crs, 
        np.column_stack((orbit_data_dict['gto']['A'], orbit_data_dict['gto']['E'], 
                        orbit_data_dict['gto']['I'], orbit_data_dict['gto']['Node'])), 
        failed_masks[1], 
        maxinc
    )
    gto_inc = gto_crs[9]
    gto_nod = gto_crs[12]

    fol_crs = np.array(fol_crs)
    fol_crs = sortdata.remove_zero_background_mag(fol_crs, background_mag_index = 21, mag_index = 20, illumination_index = 19)
    fol_crs = sortdata.sort_for_inclination_all_data(fol_crs, 9, 22)
    fol_crs, orbit_data_dict['fol']['I'], orbit_data_dict['fol']['Node'] = sortdata.inclination_filter(
        fol_crs, 
        np.column_stack((orbit_data_dict['fol']['A'], orbit_data_dict['fol']['E'], 
                        orbit_data_dict['fol']['I'], orbit_data_dict['fol']['Node'])), 
        failed_masks[2], 
        maxinc
    )
    fol_inc = fol_crs[9]
    fol_nod = fol_crs[12]

    print(f"GEO dataset size: CRS={len(geo_inc)}, Celmech={np.array(orbit_data_dict['geo']['I']).shape}")
    print(f"GTO dataset size: CRS={len(gto_inc)}, Celmech={np.array(orbit_data_dict['gto']['I']).shape}")
    print(f"FOL dataset size: CRS={len(fol_inc)}, Celmech={np.array(orbit_data_dict['fol']['I']).shape}")
    
    #plots the Celmech circular orbits and the elliptcal orbits from the crossings in one plot
    plotting.i_omega_joined(geo_nod, orbit_data_dict['geo']['Node'], geo_inc, orbit_data_dict['geo']['I'], f"Comparison Crossings to Celmech data, GEO, {year}", year, "crossings", "celmech", dir)
    plotting.i_omega_joined(gto_nod, orbit_data_dict['gto']['Node'], gto_inc, orbit_data_dict['gto']['I'], f"Comparison Crossings to Celmech data, GTO, {year}",year, "crossings", "celmech", dir)
    plotting.i_omega_joined(fol_nod, orbit_data_dict['fol']['Node'], fol_inc, orbit_data_dict['fol']['I'], f"Comparison Crossings to Celmech data, FOL, {year}", year, "crossings", "celmech", dir)    
    
    # Count failures (1s) and successes (0s) per orbit type
    for i, orbit in enumerate(orbit_type_list):
        failed_masks[i] = np.array(failed_masks[i])
        num_failures = np.sum(failed_masks[i] == 1)  # Count 1s
        num_successes = np.sum(failed_masks[i] == 0)  # Count 0s
        total_attempts = len(failed_masks[i])

        print(f"{orbit.upper()} - Total: {total_attempts}, Successes: {num_failures}, Failures: {num_successes}")
        
#Filtering: special inclination filter and background magnitude == 0 removed from crs.