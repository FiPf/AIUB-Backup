import numpy as np
import unique_objects
import sortdata
import getdata 
import plotting
import calculations
import os

def main_celmech(year:str, err: bool, ell: bool):
    """for a given year, loops over all orbit types, gets orbital elements (circular) from desired Celmech OUT file, 
    gets orbital elements (elliptical) from desired *.det file. Plots the two datasets in an I- Omega Plot (seperately and joined).

    Args:
        year (str): year of the data
        err (bool): whether the OUT file should contain noise. Noise levels are specified in the input of Celmech. 
        Note that adding noise does not make sense, so this should be to False. 
    """
    orbit_type_list = ["geo", "gto", "fol"]
    total_num_of_objects = 0
    
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

        assert len(orbit_data_dict[orbit]["I"]) == len(orbit_data_dict[orbit]["Node"]), "Filtering went wrong"   

        apogee = orbit_data_dict[orbit]["A"] * (1 + orbit_data_dict[orbit]["E"])
        mask_apogee = apogee >= apogee_threshold
        #orbit_data_dict[orbit]["I"] = orbit_data_dict[orbit]["I"][mask_apogee]
        #orbit_data_dict[orbit]["Node"] = orbit_data_dict[orbit]["Node"][mask_apogee]

    out_dir = os.path.join("output_celmech", "Plots")
    
    geo_crs, gto_crs, fol_crs, geo_det, gto_det, fol_det = getdata.data_returner(year, 4)

    
    # Sorting for geo orbit crossings
    geo_crs = np.array(geo_crs)
    
    geo_crs = sortdata.sort_for_inclination_all_data(geo_crs, 9, 22)
    geo_inc = geo_crs[9]
    geo_nod = geo_crs[12]

    # Sorting for gto orbit crossings
    gto_crs = np.array(gto_crs)
    
    gto_crs = sortdata.sort_for_inclination_all_data(gto_crs, 9, 22)
    gto_inc = gto_crs[9]
    gto_nod = gto_crs[12]

    # Sorting for fol orbit crossings
    fol_crs = np.array(fol_crs)   
    
    fol_crs = sortdata.sort_for_inclination_all_data(fol_crs, 9, 22)
    fol_inc = fol_crs[9]
    fol_nod = fol_crs[12]
    
    tolerance = [2000, 10, 0.1, 0.1]
    #order: a, e, i, Node
    #divide celmech semi major axes by 1000 to get matching units for comparison
    celmech_data_geo = np.column_stack([orbit_data_dict["geo"]["A"]/1000, orbit_data_dict["geo"]["E"], orbit_data_dict["geo"]["I"], orbit_data_dict["geo"]["Node"]])
    celmech_data_gto = np.column_stack([orbit_data_dict["gto"]["A"]/1000, orbit_data_dict["gto"]["E"], orbit_data_dict["gto"]["I"], orbit_data_dict["gto"]["Node"]])
    celmech_data_fol = np.column_stack([orbit_data_dict["fol"]["A"]/1000, orbit_data_dict["fol"]["E"], orbit_data_dict["fol"]["I"], orbit_data_dict["fol"]["Node"]])

    crs_data_geo = np.column_stack([geo_crs[8], geo_crs[10], geo_crs[9], geo_crs[12]])
    crs_data_gto = np.column_stack([gto_crs[8], gto_crs[10], gto_crs[9], gto_crs[12]])
    crs_data_fol = np.column_stack([fol_crs[8], fol_crs[10], fol_crs[9], fol_crs[12]])
    
    celmech_mask_geo = failed_masks[0]
    celmech_mask_gto = failed_masks[1]
    celmech_mask_fol = failed_masks[2]
    
    matches_geo, unmatched_geo, unmatched_crs_data_geo = compare_orbital_elements(celmech_data_geo, crs_data_geo, celmech_mask_geo, tolerance)
    matches_gto, unmatched_gto, unmatched_crs_data_gto = compare_orbital_elements(celmech_data_gto, crs_data_gto, celmech_mask_gto, tolerance)
    matches_fol, unmatched_fol, unmatched_crs_data_fol = compare_orbital_elements(celmech_data_fol, crs_data_fol, celmech_mask_fol, tolerance)

    # Plotting matched GEO data
    matches_geo_nodes, matches_geo_inclinations, matches_geo_ecc, crs_geo_nodes, crs_geo_inclinations, crs_geo_ecc = extract_matched_data(matches_geo, "geo")

    plotting.i_omega_joined_with_eccentricity(
        matches_geo_nodes, crs_geo_nodes, matches_geo_inclinations, crs_geo_inclinations, matches_geo_ecc, crs_geo_ecc,
        f"Simulations: GEO circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )
    
    plotting.i_omega_joined(
        matches_geo_nodes, crs_geo_nodes, matches_geo_inclinations, crs_geo_inclinations,
        f"Simulations: GEO circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )

    # Show results for GEO matches
    print("Results GEO Matches")
    print(f"Matched: {matches_geo}")
    print(f"Unmatched: {unmatched_geo}")
    print(f"Match percentage: {len(matches_geo)/(len(matches_geo) + len(unmatched_geo))}")

    # Plotting matched GTO data
    matches_gto_nodes, matches_gto_inclinations, matches_gto_ecc, crs_gto_nodes, crs_gto_inclinations, crs_gto_ecc = extract_matched_data(matches_gto, "gto")

    plotting.i_omega_joined_with_eccentricity(
        matches_gto_nodes, crs_gto_nodes, matches_gto_inclinations, crs_gto_inclinations, matches_gto_ecc, crs_gto_ecc,
        f"Simulations: GTO circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )
    
    plotting.i_omega_joined(
        matches_gto_nodes, crs_gto_nodes, matches_gto_inclinations, crs_gto_inclinations,
        f"Simulations: GTO circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )

    # Show results for GTO matches
    print("Results GTO Matches")
    print(f"Matched: {matches_gto}")
    print(f"Unmatched: {unmatched_gto}")
    print(f"Match percentage: {len(matches_gto)/(len(matches_gto) + len(unmatched_gto))}")

    # Plotting matched FollowUp data
    matches_fol_nodes, matches_fol_inclinations, matches_fol_ecc, crs_fol_nodes, crs_fol_inclinations, crs_fol_ecc = extract_matched_data(matches_fol, "fol")

    plotting.i_omega_joined_with_eccentricity(
        matches_fol_nodes, crs_fol_nodes, matches_fol_inclinations, crs_fol_inclinations, matches_fol_ecc, crs_fol_ecc,
        f"Simulations: FollowUp circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )
    
    plotting.i_omega_joined(
        matches_fol_nodes, crs_fol_nodes, matches_fol_inclinations, crs_fol_inclinations,
        f"Simulations: FollowUp circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )

    # Show results for FollowUp matches
    print("Results FollowUp Matches")
    print(f"Matched: {matches_fol}")
    print(f"Unmatched: {unmatched_fol}")
    print(f"Match percentage: {len(matches_gto)/(len(matches_gto) + len(unmatched_gto))}")

    # Extract unmatched data for GEO
    unmatched_geo_nodes, unmatched_geo_inclinations, unmatched_geo_ecc, crs_unmatched_geo_nodes, crs_unmatched_geo_inclinations, crs_unmatched_geo_ecc = extract_unmatched_data(unmatched_geo, unmatched_crs_data_geo, "geo")

    # Plot unmatched GEO data
    plotting.i_omega_joined_with_eccentricity(
        unmatched_geo_nodes, crs_unmatched_geo_nodes, unmatched_geo_inclinations, crs_unmatched_geo_inclinations, unmatched_geo_ecc, crs_unmatched_geo_ecc,
        f"Simulations: GEO unmatched circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )
    
    plotting.i_omega_joined(
        unmatched_geo_nodes, crs_unmatched_geo_nodes, unmatched_geo_inclinations, crs_unmatched_geo_inclinations,
        f"Simulations: GEO unmatched circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )

    # Extract unmatched data for GTO
    unmatched_gto_nodes, unmatched_gto_inclinations, unmatched_gto_ecc, crs_unmatched_gto_nodes, crs_unmatched_gto_inclinations, crs_unmatched_gto_ecc = extract_unmatched_data(unmatched_gto, unmatched_crs_data_gto, "gto")

    # Plot unmatched GTO data
    plotting.i_omega_joined_with_eccentricity(
        unmatched_gto_nodes, crs_unmatched_gto_nodes, unmatched_gto_inclinations, crs_unmatched_gto_inclinations, unmatched_gto_ecc, crs_unmatched_gto_ecc,
        f"Simulations: GTO unmatched circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )

    plotting.i_omega_joined(
        unmatched_gto_nodes, crs_unmatched_gto_nodes, unmatched_gto_inclinations, crs_unmatched_gto_inclinations,
        f"Simulations: GTO unmatched circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )

    # Extract unmatched data for FOL
    unmatched_fol_nodes, unmatched_fol_inclinations, unmatched_fol_ecc, crs_unmatched_fol_nodes, crs_unmatched_fol_inclinations, crs_unmatched_fol_ecc = extract_unmatched_data(unmatched_fol, unmatched_crs_data_fol, "fol")

    # Plot unmatched FOL data
    plotting.i_omega_joined_with_eccentricity(
        unmatched_fol_nodes, crs_unmatched_fol_nodes, unmatched_fol_inclinations, crs_unmatched_fol_inclinations, unmatched_fol_ecc, crs_unmatched_fol_ecc,
        f"Simulations: FOL unmatched circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )
    
    plotting.i_omega_joined(
        unmatched_fol_nodes, crs_unmatched_fol_nodes, unmatched_fol_inclinations, crs_unmatched_fol_inclinations,
        f"Simulations: FOL unmatched circular vs. elliptical orbits {year}" + title_appendix,
        year, "circular", "elliptical", out_dir
    )

def compare_orbital_elements(celmech_data: np.array, crs_data: np.array, celmech_mask: np.array, tolerance: np.array):
    """
    Compare orbital elements from Celmech and CRS data row by row.
    If a match is found, store the rows in a data structure with all values.
    If no match is found, check the mask and log the failure.

    Args:
        celmech_data (np.array): Celmech orbital elements as a 2D array (rows are objects).
                                 Columns: [a, e, i, Node]
        crs_data (np.array): CRS orbital elements as a 2D array (rows are objects).
                             Columns: [a, e, i, Node]
        celmech_mask (np.array): Mask indicating success (1) or failure (0) for Celmech rows.
        tolerance (np.array): Tolerance array for matching orbital elements.

    Returns:
        matches (list): List of dictionaries containing matched rows and their details.
        unmatched (list): List of Celmech rows with no match in CRS data.
        unmatched_crs_data (np.array): CRS rows that did not match any Celmech data.
    """
    matches = []
    unmatched = []
    matched_crs_indices = set()

    # Iterate over Celmech rows
    for i, (a_c, e_c, i_c, node_c) in enumerate(celmech_data):
        # Skip rows with a mask of 0
        if celmech_mask[i] == 0:
            unmatched.append({
                "celmech_row": [a_c, e_c, i_c, node_c],
                "crs_row": None,  # No CRS row to match
                "celmech_index": i,
                "crs_index": None,
                "reason": "mask_failure"
            })
            continue

        # Flag to track if a match is found
        match_found = False

        # Compare with each row in CRS data
        for j, (a_r, e_r, i_r, node_r) in enumerate(crs_data):
            if (abs(a_c - a_r) <= tolerance[0] and
                abs(e_c - e_r) <= tolerance[1] and
                abs(i_c - i_r) <= tolerance[2] and
                abs(node_c - node_r) <= tolerance[3]):
                # Match found
                matches.append({
                    "celmech_row": [a_c, e_c, i_c, node_c],
                    "crs_row": [a_r, e_r, i_r, node_r],
                    "celmech_index": i,
                    "crs_index": j
                })
                matched_crs_indices.add(j)  # Add matched CRS index to the set
                match_found = True
                break

        # If no match was found, log the Celmech row
        if not match_found:
            unmatched.append({
                "celmech_row": [a_c, e_c, i_c, node_c],
                "crs_row": None,  # No CRS row to match
                "celmech_index": i,
                "crs_index": None,
                "reason": "no_match"
            })

    # Filter out the matched rows from CRS data
    unmatched_crs_data = [crs_data[i] for i in range(len(crs_data)) if i not in matched_crs_indices]

    return matches, unmatched, unmatched_crs_data

# Function to extract matched data for plotting
def extract_matched_data(matches, data_type):
    """Extract matched data for plotting from the 'matches' list."""
    celmech_nodes = []
    celmech_inclinations = []
    celmech_ecc = []
    crs_nodes = []
    crs_inclinations = []
    crs_ecc = []
    
    for match in matches:
        if data_type == "geo":
            celmech_nodes.append(match["celmech_row"][3])  # Node for Celmech (GEO)
            celmech_inclinations.append(match["celmech_row"][2])  # Inclination for Celmech (GEO)
            celmech_ecc.append(match["celmech_row"][1])
            crs_nodes.append(match["crs_row"][3])  # Node for CRS (GEO)
            crs_inclinations.append(match["crs_row"][2])  # Inclination for CRS (GEO)
            crs_ecc.append(match["crs_row"][1])
            
        elif data_type == "gto":
            celmech_nodes.append(match["celmech_row"][3])  # Node for Celmech (GTO)
            celmech_inclinations.append(match["celmech_row"][2])  # Inclination for Celmech (GTO)
            celmech_ecc.append(match["celmech_row"][1])
            crs_nodes.append(match["crs_row"][3])  # Node for CRS (GTO)
            crs_inclinations.append(match["crs_row"][2])  # Inclination for CRS (GTO)
            crs_ecc.append(match["crs_row"][1])

        elif data_type == "fol":
            celmech_nodes.append(match["celmech_row"][3])  # Node for Celmech (FollowUp)
            celmech_inclinations.append(match["celmech_row"][2])  # Inclination for Celmech (FollowUp)
            celmech_ecc.append(match["celmech_row"][1])
            crs_nodes.append(match["crs_row"][3])  # Node for CRS (FollowUp)
            crs_inclinations.append(match["crs_row"][2])  # Inclination for CRS (FollowUp)
            crs_ecc.append(match["crs_row"][1])

    return celmech_nodes, celmech_inclinations, celmech_ecc, crs_nodes, crs_inclinations, crs_ecc

def extract_unmatched_data(unmatched_data, unmatched_crs_data, data_type):
    """Extract unmatched data for plotting from the 'unmatched_data' list."""
    celmech_nodes = []
    celmech_inclinations = []
    celmech_ecc = []
    crs_nodes = []
    crs_inclinations = []
    crs_ecc = []
    
    # Create a dictionary for easy look-up of unmatched CRS data by index
    unmatched_crs_dict = {i: row for i, row in enumerate(unmatched_crs_data)}

    for match in unmatched_data:
        # Append Celmech data (all orbit types)
        celmech_nodes.append(match["celmech_row"][3])  # Node for Celmech
        celmech_inclinations.append(match["celmech_row"][2])  # Inclination for Celmech
        celmech_ecc.append(match["celmech_row"][1])
        
        # Handle different orbit types (geo, gto, fol)
        if data_type == "geo":
            # If no match, use CRS data from unmatched_crs_data
            if match["crs_row"] is None:
                index = match["celmech_index"]
                crs_row = unmatched_crs_dict.get(index, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                crs_nodes.append(crs_row[3])  # Node for CRS (GEO)
                crs_inclinations.append(crs_row[2])  # Inclination for CRS (GEO)
                crs_ecc.append(crs_row[1])
            else:
                crs_nodes.append(match["crs_row"][3])  # Node for CRS (GEO)
                crs_inclinations.append(match["crs_row"][2])  # Inclination for CRS (GEO)
                crs_ecc.append(match["crs_row"][1])

        elif data_type == "gto":
            # If no match, use CRS data from unmatched_crs_data
            if match["crs_row"] is None:
                index = match["celmech_index"]
                crs_row = unmatched_crs_dict.get(index, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                crs_nodes.append(crs_row[3])  # Node for CRS (GTO)
                crs_inclinations.append(crs_row[2])  # Inclination for CRS (GTO)
                crs_ecc.append(crs_row[1])
            else:
                crs_nodes.append(match["crs_row"][3])  # Node for CRS (GTO)
                crs_inclinations.append(match["crs_row"][2])  # Inclination for CRS (GTO)
                crs_ecc.append(match["crs_row"][1])

        elif data_type == "fol":
            # If no match, use CRS data from unmatched_crs_data
            if match["crs_row"] is None:
                index = match["celmech_index"]
                crs_row = unmatched_crs_dict.get(index, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                crs_nodes.append(crs_row[3])  # Node for CRS (FollowUp)
                crs_inclinations.append(crs_row[2])  # Inclination for CRS (FollowUp)
                crs_ecc.append(crs_row[1])
            else:
                crs_nodes.append(match["crs_row"][3])  # Node for CRS (FollowUp)
                crs_inclinations.append(match["crs_row"][2])  # Inclination for CRS (FollowUp)
                crs_ecc.append(match["crs_row"][1])

    return celmech_nodes, celmech_inclinations, celmech_ecc, crs_nodes, crs_inclinations, crs_ecc