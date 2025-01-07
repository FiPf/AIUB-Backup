import numpy as np
import unique_objects
import sortdata
import getdata 
import plotting
import calculations
import os

def main_celmech(year:str, err: bool):
    """for a given year, loops over all orbit types, gets orbital elements (circular) from desired Celmech OUT file, 
    gets orbital elements (elliptical) from desired *.det file. Plots the two datasets in an I- Omega Plot (seperately and joined).

    Args:
        year (str): year of the data
        err (bool): whether the OUT file should contain noise. Noise levels are specified in the input of Celmech. 
        Note that adding noise does not make sense, so this should be to False. 
    """
    orbit_type_list = ["geo", "gto", "fol"]
    total_num_of_objects = 0
    yy = year[2:]
    
    #STEP 1: Unpack plugin data and store the epochs in a dictionary
    plugin_data_dict = {
        "geo": {"epoch": []}, 
        "gto": {"epoch": []}, 
        "fol": {"epoch": []}
    }
    
    for orbit in orbit_type_list: 
        file = f"plugin_{yy}_{orbit}.pro" 
        filename = os.path.join("input", file)
        plugin_data = getdata.array_extender_plugin(filename)
        epoch = plugin_data[2]
        plugin_data_dict[orbit]["epoch"].append(epoch)
    
    epochs_plugin = np.hstack([np.array(plugin_data_dict["geo"]["epoch"]), np.array(plugin_data_dict["gto"]["epoch"]), np.array(plugin_data_dict["fol"]["epoch"])])
    
    #STEP 2: Unpack Celmech output data
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
        file = getdata.get_celmech_OUT_files(year, orbit, err, ell = False) #find the right Celmech file for given year and orbit type
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

        assert len(orbit_data_dict[orbit]["I"]) == len(orbit_data_dict[orbit]["Node"]), "Filtering went wrong"   

        apogee = orbit_data_dict[orbit]["A"] * (1 + orbit_data_dict[orbit]["E"])
        mask_apogee = apogee >= apogee_threshold
        orbit_data_dict[orbit]["I"] = orbit_data_dict[orbit]["I"][mask_apogee]
        orbit_data_dict[orbit]["Node"] = orbit_data_dict[orbit]["Node"][mask_apogee]

    out_dir = os.path.join("output_celmech", "Plots")

    """plotting.i_omega_all_orbits(
        orbit_data_dict["geo"]["Node"], orbit_data_dict["gto"]["Node"], orbit_data_dict["fol"]["Node"], 
        orbit_data_dict["geo"]["I"], orbit_data_dict["gto"]["I"], orbit_data_dict["fol"]["I"], 
        f"Simulations: circular orbits {year}" + title_appendix, year, out_dir)"""

    print(f"Number of objects in the plot from celmech: {len(orbit_data_dict["geo"]["Node"]) + len(orbit_data_dict["gto"]["Node"]) + len(orbit_data_dict["fol"]["Node"])}")
    #print(f"Number of geo objects Celmech: {len(orbit_data_dict['geo']['Node'])}")
    #print(f"Number of gto objects Celmech: {len(orbit_data_dict['gto']['Node'])}")
    #print(f"Number of fol objects Celmech: {len(orbit_data_dict['fol']['Node'])}")

    #STEP 3: Unpack *.crs data from PROOF output
    # Data handling for elliptical orbits from PROOF *.crs and *.det files 
    geo_crs, gto_crs, fol_crs, geo_det, gto_det, fol_det = getdata.data_returner(year, 4)
    
    #print(f"crs geo: {len(geo_crs[1])}, crs gto {len(gto_crs[1])}, crs fol {len(fol_crs[1])},  before any sorting")
    #print("Sum of crossings:", len(geo_crs[1]) + len(gto_crs[1]) + len(fol_crs[1]))
    #print(len(geo_det[1]), len(gto_det[1]), len(fol_det[1]), "detections before any sorting")
    #print("Sum of detections:", len(geo_det[1]) + len(gto_det[1]) + len(fol_det[1]))

    ID = np.hstack([geo_crs[0], gto_crs[0], fol_crs[0]]) #stack all ID arrays
    unique_ID = set(ID)
    #print("Number of all IDs: ", len(ID),  ", unique ID crossings no sorting", len(unique_ID))
    
    #ID = np.hstack([geo_det[0], gto_det[0], fol_det[0]]) #stack all ID arrays
    #unique_ID = set(ID)
    #print("Number of all IDs: ", len(ID),  "unique ID detections no sorting", len(unique_ID))

    # Sorting for geo orbit crossings
    geo_crs = np.array(geo_crs)
    geo_crs = geo_crs[:, :len(failed_masks[0])]
    mask = np.array(failed_masks[0], dtype=bool).flatten()
    geo_crs = geo_crs[:, mask]

    #remove background zero objects from *.crs file data
    geo_crs = sortdata.remove_zero_background_mag(geo_crs, background_mag_index = 21)
    print(len(geo_crs[1]), "Number of crossings after background")
    geo_det = sortdata.remove_zero_background_mag(geo_det, background_mag_index = 21)
    print(len(geo_det[1]), "Number of detections after background")
    TLE, fragments, rest = sortdata.sort_for_sources_all_data(geo_crs, 3)
    arrays_to_stack = []
    for arr in [TLE, fragments, rest]:
        if arr.size > 0: 
            arrays_to_stack.append(arr)

    if arrays_to_stack:
        geo_crs = np.vstack(arrays_to_stack).T
    else:
        geo_crs = np.empty((geo_crs.shape[0], 0))  
    geo_crs = sortdata.sort_for_apogee_all_data(geo_crs, 10, 8)
    geo_crs = sortdata.sort_for_inclination_all_data(geo_crs, 9, 22)
    geo_inc = geo_crs[9]
    geo_nod = geo_crs[12]
    geo_TCA = geo_crs[4] #time of closest approach

    # Sorting for gto orbit crossings
    gto_crs = np.array(gto_crs)
    gto_crs = gto_crs[:, :len(failed_masks[1])]
    mask = np.array(failed_masks[1], dtype=bool).flatten()
    gto_crs = gto_crs[:, mask]
    
    #remove background zero objects from *.crs file data
    gto_crs = sortdata.remove_zero_background_mag(gto_crs, background_mag_index = 21)
    #print(len(gto_crs[1]), "Number of crossings after background")
    gto_det = sortdata.remove_zero_background_mag(gto_det, background_mag_index = 21)
    #print(len(gto_det[1]), "Number of detections after background")
    TLE, fragments, rest = sortdata.sort_for_sources_all_data(gto_crs, 3)    
    arrays_to_stack = []
    for arr in [TLE, fragments, rest]:
        if arr.size > 0:  
            arrays_to_stack.append(arr)
    if arrays_to_stack:
        gto_crs = np.vstack(arrays_to_stack).T 
    else:
        gto_crs = np.empty((gto_crs.shape[0], 0))  
    
    gto_crs = sortdata.sort_for_apogee_all_data(gto_crs, 10, 8)
    gto_crs = sortdata.sort_for_inclination_all_data(gto_crs, 9, 22)
    gto_inc = gto_crs[9]
    gto_nod = gto_crs[12]
    gto_TCA = gto_crs[4] #time of closest approach

    # Sorting for fol orbit crossings
    fol_crs = np.array(fol_crs)   
    fol_crs = fol_crs[:, :len(failed_masks[2])]
    mask = np.array(failed_masks[2], dtype=bool).flatten()
    fol_crs = fol_crs[:, mask] 

    #remove background zero objects from *.crs file data
    fol_crs = sortdata.remove_zero_background_mag(fol_crs, background_mag_index = 21)
    #print(len(fol_crs[1]), "Number of crossings after background")
    fol_det = sortdata.remove_zero_background_mag(fol_det, background_mag_index = 21)
    #print(len(fol_det[1]), "Number of detections after background")
    common_elements = np.isin(fol_det[1], fol_crs[1], True)
    #print(len(common_elements), "number of common elements")
    TLE, fragments, rest = sortdata.sort_for_sources_all_data(fol_crs, 3)
    arrays_to_stack = []
    for arr in [TLE, fragments, rest]:
        if arr.size > 0:  
            arrays_to_stack.append(arr)

    if arrays_to_stack:
        fol_crs = np.vstack(arrays_to_stack).T 
    else:
        fol_crs = np.empty((fol_crs.shape[0], 0))  
    fol_crs = sortdata.sort_for_apogee_all_data(fol_crs, 10, 8)
    fol_crs = sortdata.sort_for_inclination_all_data(fol_crs, 9, 22)
    fol_inc = fol_crs[9]
    fol_nod = fol_crs[12]
    fol_TCA = fol_crs[4] #time of closest approach
        
    #print(len(geo_crs[1]), len(gto_crs[1]), len(fol_crs[1]), "crossings after sorting")
    #print(f"Sum of crossings after sorting: {len(geo_crs[1]) + len(gto_crs[1]) + len(fol_crs[1])}")

    # Plot elliptical orbits
    """plotting.i_omega_all_orbits(
        geo_nod, gto_nod, fol_nod, geo_inc, gto_inc, fol_inc, 
        f"Simulations: elliptical orbits {year}" + title_appendix, year, out_dir)"""
    
    #STEP 4: Compare *.crs and Celmech output dates to find matches
    epochs_crs = np.hstack([geo_TCA, gto_TCA, fol_TCA])
    epochs_crs = calculations.convert_TCA_to_mjd(epochs_crs)
    epochs_celmech = dates
    epochs_plugin = epochs_plugin
    print("epochs crs", epochs_crs)
    print("epochs celmech", epochs_celmech)
    print("crs", len(epochs_crs), "celmech", len(epochs_celmech), "plugin", epochs_plugin.shape)
    matches = calculations.find_matching_indices_MJD(epochs_crs, epochs_celmech)
    
    differences_min = [(crs - cel) * 60 * 24 for crs, cel in zip(epochs_crs, epochs_celmech)]
    print("differences")
    print(differences_min)
    
    valid_matches = [match for match in matches if match[1] is not None and match[0] is not None]
    print("Matches")
    print(matches)
    print("valid")
    print(np.array(valid_matches).shape)
    print(len(epochs_crs), len(epochs_celmech), len(matches))
    
    valid_crs_indices = [match[0] for match in valid_matches]
    valid_celmech_indices = [match[1] for match in valid_matches]
    
    print(len(valid_crs_indices), len(valid_celmech_indices)) 
    
    if not valid_celmech_indices or not valid_crs_indices:
        raise IndexError("No valid indices found after filtering.")
    
    #STEP: Plotting the final results
    # Plot circular and elliptical orbits together
    nod_circ = np.hstack([orbit_data_dict["geo"]["Node"], orbit_data_dict["gto"]["Node"], orbit_data_dict["fol"]["Node"]])
    i_circ = np.hstack([orbit_data_dict["geo"]["I"], orbit_data_dict["gto"]["I"], orbit_data_dict["fol"]["I"]])
    
    nod_ell = np.hstack([geo_nod, gto_nod, fol_nod])
    i_ell = np.hstack([geo_inc, gto_inc, fol_inc])
    
    valid_celmech_indices = [index for index in valid_celmech_indices if index < len(nod_circ)]
    valid_crs_indices = [index for index in valid_crs_indices if index < len(nod_ell)]
    print(nod_circ.shape, i_circ.shape, nod_ell.shape, i_ell.shape) 
    
    plotting.i_omega_joined(
        nod_circ[valid_celmech_indices], nod_ell[valid_crs_indices], i_circ[valid_celmech_indices], i_ell[valid_crs_indices], 
        f"Simulations: circular vs. elliptical orbits {year}" + title_appendix, 
        year, "circular", "elliptical", out_dir)    
    #print(geo_TCA)