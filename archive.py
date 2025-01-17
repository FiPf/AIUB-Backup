#this is horrible design
def data_for_one_year_one_seed_th25(year: str, seed: str):
    """used to get all the simulation data for a specific year and seed, not separated/sorted

    Args:
        year (str): year of data
        seed (str): seed of data (1, 2, 3 or 4)

    Returns:
        data_crs (np.array): crossing data for that year
        data_det (np.array): detection data for that year
    """
    year2 = year[2:]

    GEO_file_crs = f"stat_Master_{year2}_geo_s{seed}_new.crs"
    GTO_file_crs = f"stat_Master_{year2}_gto_s{seed}_new.crs"
    followup_file_crs = f"stat_Master_{year2}_fol_s{seed}_new.crs"
    GEO_file_crs = os.path.join("input", GEO_file_crs)
    GTO_file_crs = os.path.join("input", GTO_file_crs)
    followup_file_crs = os.path.join("input", followup_file_crs)

    data_GTO_crs = array_extender(GTO_file_crs)
    data_GEO_crs = array_extender(GEO_file_crs)
    data_followup_crs = array_extender(followup_file_crs)

    data_crs = np.hstack([data_GEO_crs, data_GTO_crs, data_followup_crs])

    GEO_file_det = f"stat_Master_{year2}_geo_s{seed}_new.det"
    GTO_file_det = f"stat_Master_{year2}_gto_s{seed}_new.det"
    followup_file_det = f"stat_Master_{year2}_fol_s{seed}_new.det"
    GEO_file_det = os.path.join("input", GEO_file_det)
    GTO_file_det = os.path.join("input", GTO_file_det)
    followup_file_det = os.path.join("input", followup_file_det)

    data_GTO_det = array_extender(GTO_file_det)
    data_GEO_det = array_extender(GEO_file_det)
    data_followup_det = array_extender(followup_file_det)
    
    data_det = np.hstack([data_GEO_det, data_GTO_det, data_followup_det])
    
    return data_crs, data_det


#this is horrible design
def data_for_one_year_one_seed_th3(year: str, seed: str):
    """used to get all the simulation data for a specific year and seed, not separated/sorted

    Args:
        year (str): year of data
        seed (str): seed of data (1, 2, 3 or 4)

    Returns:
        data_crs (np.array): crossing data for that year
        data_det (np.array): detection data for that year
    """
    year2 = year[2:]

    GEO_file_crs = f"stat_Master_{year2}_geo_s{seed}_new2.crs"
    GTO_file_crs = f"stat_Master_{year2}_gto_s{seed}_new2.crs"
    followup_file_crs = f"stat_Master_{year2}_fol_s{seed}_new2.crs"
    GEO_file_crs = os.path.join("input", GEO_file_crs)
    GTO_file_crs = os.path.join("input", GTO_file_crs)
    followup_file_crs = os.path.join("input", followup_file_crs)

    data_GTO_crs = array_extender(GTO_file_crs)
    data_GEO_crs = array_extender(GEO_file_crs)
    data_followup_crs = array_extender(followup_file_crs)

    data_crs = np.hstack([data_GEO_crs, data_GTO_crs, data_followup_crs])

    GEO_file_det = f"stat_Master_{year2}_geo_s{seed}_new2.det"
    GTO_file_det = f"stat_Master_{year2}_gto_s{seed}_new2.det"
    followup_file_det = f"stat_Master_{year2}_fol_s{seed}_new2.det"
    GEO_file_det = os.path.join("input", GEO_file_det)
    GTO_file_det = os.path.join("input", GTO_file_det)
    followup_file_det = os.path.join("input", followup_file_det)

    data_GTO_det = array_extender(GTO_file_det)
    data_GEO_det = array_extender(GEO_file_det)
    data_followup_det = array_extender(followup_file_det)
    
    data_det = np.hstack([data_GEO_det, data_GTO_det, data_followup_det])
    
    return data_crs, data_det

#even more horrible design
def data_for_one_year_one_seed_newpop(year: str, seed: str):
    """used to get all the simulation data for a specific year and seed, not separated/sorted
    This function is used for the simulations with the "new" population files. At the beginning, the most recent
    population file was from 2016, all simulations for later years were performed with this. Now, we have 
    yearly population files for up to 2024, the simulations for those years were repeated. 

    Args:
        year (str): year of data
        seed (str): seed of data (1, 2, 3 or 4)

    Returns:
        data_crs (np.array): crossing data for that year
        data_det (np.array): detection data for that year
    """
    year2 = year[2:]
    
    if int(year2) < 18: 
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

    else: 
        GEO_file_crs = f"stat_Master_{year2}_geo_s{seed}_newpo.crs"
        GTO_file_crs = f"stat_Master_{year2}_gto_s{seed}_newpo.crs"
        followup_file_crs = f"stat_Master_{year2}_fol_s{seed}_newpo.crs"
        GEO_file_crs = os.path.join("input", GEO_file_crs)
        GTO_file_crs = os.path.join("input", GTO_file_crs)
        followup_file_crs = os.path.join("input", followup_file_crs)

        data_GTO_crs = array_extender(GTO_file_crs)
        data_GEO_crs = array_extender(GEO_file_crs)
        data_followup_crs = array_extender(followup_file_crs)

        data_crs = np.hstack([data_GEO_crs, data_GTO_crs, data_followup_crs])

        GEO_file_det = f"stat_Master_{year2}_geo_s{seed}_newpo.det"
        GTO_file_det = f"stat_Master_{year2}_gto_s{seed}_newpo.det"
        followup_file_det = f"stat_Master_{year2}_fol_s{seed}_newpo.det"
        GEO_file_det = os.path.join("input", GEO_file_det)
        GTO_file_det = os.path.join("input", GTO_file_det)
        followup_file_det = os.path.join("input", followup_file_det)

        data_GTO_det = array_extender(GTO_file_det)
        data_GEO_det = array_extender(GEO_file_det)
        data_followup_det = array_extender(followup_file_det)
        
        data_det = np.hstack([data_GEO_det, data_GTO_det, data_followup_det])
    
    return data_crs, data_det

def data_for_one_year_one_seed_newpop_th3(year: str, seed: str):
    """used to get all the simulation data for a specific year and seed, not separated/sorted
    This function is used for the simulations with the "new" population files. At the beginning, the most recent
    population file was from 2016, all simulations for later years were performed with this. Now, we have 
    yearly population files for up to 2024, the simulations for those years were repeated. 

    Args:
        year (str): year of data
        seed (str): seed of data (1, 2, 3 or 4)

    Returns:
        data_crs (np.array): crossing data for that year
        data_det (np.array): detection data for that year
    """
    year2 = year[2:]
    
    if int(year2) < 18: 
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

    else: 
        GEO_file_crs = f"stat_Master_{year2}_geo_s{seed}_npth3.crs"
        GTO_file_crs = f"stat_Master_{year2}_gto_s{seed}_npth3.crs"
        followup_file_crs = f"stat_Master_{year2}_fol_s{seed}_npth3.crs"
        GEO_file_crs = os.path.join("input", GEO_file_crs)
        GTO_file_crs = os.path.join("input", GTO_file_crs)
        followup_file_crs = os.path.join("input", followup_file_crs)

        data_GTO_crs = array_extender(GTO_file_crs)
        data_GEO_crs = array_extender(GEO_file_crs)
        data_followup_crs = array_extender(followup_file_crs)

        data_crs = np.hstack([data_GEO_crs, data_GTO_crs, data_followup_crs])

        GEO_file_det = f"stat_Master_{year2}_geo_s{seed}_npth3.det"
        GTO_file_det = f"stat_Master_{year2}_gto_s{seed}_npth3.det"
        followup_file_det = f"stat_Master_{year2}_fol_s{seed}_npth3.det"
        GEO_file_det = os.path.join("input", GEO_file_det)
        GTO_file_det = os.path.join("input", GTO_file_det)
        followup_file_det = os.path.join("input", followup_file_det)

        data_GTO_det = array_extender(GTO_file_det)
        data_GEO_det = array_extender(GEO_file_det)
        data_followup_det = array_extender(followup_file_det)
        
        data_det = np.hstack([data_GEO_det, data_GTO_det, data_followup_det])
    
    return data_crs, data_det

#worst design
def data_for_one_year_one_seed_propagate(year: str, seed: str):
    """used to get all the simulation data for a specific year and seed, not separated/sorted
    This function is used for the simulations with the "new" population files. At the beginning, the most recent
    population file was from 2016, all simulations for later years were performed with this. Now, we have 
    yearly population files for up to 2024, the simulations for those years were repeated. 

    Args:
        year (str): year of data
        seed (str): seed of data (1, 2, 3 or 4)

    Returns:
        data_crs (np.array): crossing data for that year
        data_det (np.array): detection data for that year
    """
    year2 = year[2:]
    
    if int(year2) < 19: 
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

    else: 
        GEO_file_crs = f"stat_Master_{year2}_geo_s{seed}_prop.crs"
        GTO_file_crs = f"stat_Master_{year2}_gto_s{seed}_prop.crs"
        followup_file_crs = f"stat_Master_{year2}_fol_s{seed}_prop.crs"
        GEO_file_crs = os.path.join("input", GEO_file_crs)
        GTO_file_crs = os.path.join("input", GTO_file_crs)
        followup_file_crs = os.path.join("input", followup_file_crs)

        data_GTO_crs = array_extender(GTO_file_crs)
        data_GEO_crs = array_extender(GEO_file_crs)
        data_followup_crs = array_extender(followup_file_crs)

        data_crs = np.hstack([data_GEO_crs, data_GTO_crs, data_followup_crs])

        GEO_file_det = f"stat_Master_{year2}_geo_s{seed}_prop.det"
        GTO_file_det = f"stat_Master_{year2}_gto_s{seed}_prop.det"
        followup_file_det = f"stat_Master_{year2}_fol_s{seed}_prop.det"
        GEO_file_det = os.path.join("input", GEO_file_det)
        GTO_file_det = os.path.join("input", GTO_file_det)
        followup_file_det = os.path.join("input", followup_file_det)

        data_GTO_det = array_extender(GTO_file_det)
        data_GEO_det = array_extender(GEO_file_det)
        data_followup_det = array_extender(followup_file_det)
        
        data_det = np.hstack([data_GEO_det, data_GTO_det, data_followup_det])
    
    return data_crs, data_det