import numpy as np
import unique_objects
import sortdata
import getdata 
import plotting
import calculations
#the main function is used to process the data for one seed and one run and get all the desired plots/numbers
#version of main using data = file directly instead of using array_extender, input is not str (file for the array extender), but np.array
#all fragments means: fragments + rest are included in the data

def main_all_fragments(crs_file: np.array, det_file: np.array, year: str, orbit_type: str, seed: str, directory: str):
    #store crossing data in arrays
    data = crs_file
    sources_crs = data[3]
    magnitudes_crs = data[20]
    inc_crs = data[9]
    omega_crs = data[12]
    diameter_crs = data[1]
    
    #only used for apogee sorting
    sem_major_crs = data[8]
    ecc_crs = data[10]
    
    #apogee sorting
    sorted = sortdata.sort_for_apogee(sem_major_crs, ecc_crs, sources_crs, magnitudes_crs, inc_crs, omega_crs, diameter_crs)
    sources_crs = sorted[0]
    magnitudes_crs = sorted[1]
    inc_crs = sorted[2]
    omega_crs = sorted[3]
    diameter_crs = sorted[4]
    
    #inclination sorting
    max_inc = 22
    sorted = sortdata.sort_for_inclination(inc_crs, max_inc, sources_crs, magnitudes_crs, omega_crs, diameter_crs)
    sources_crs = sorted[0]
    magnitudes_crs = sorted[1]
    inc_crs = [i for i in inc_crs if i < max_inc]
    omega_crs = sorted[2]
    diameter_crs = sorted[3]
    
    #store detected data in arrays
    data = det_file
    sources_det = data[3]
    magnitudes_det = data[20]
    inc_det = data[9]
    omega_det = data[12]
    diameter_det = data[1]
    
    #only used for apogee sorting
    sem_major_det = data[8]
    ecc_det = data[10]

    #apogee sorting
    sorted = sortdata.sort_for_apogee(sem_major_det, ecc_det, sources_det, magnitudes_det, inc_det, omega_det, diameter_det)
    sources_det = sorted[0]
    magnitudes_det = sorted[1]
    inc_det = sorted[2]
    omega_det = sorted[3]
    diameter_det = sorted[4]
    
    #inclination sorting
    max_inc = 22
    sorted = sortdata.sort_for_inclination(inc_det, max_inc, sources_det, magnitudes_det, omega_det, diameter_det)
    sources_det= sorted[0]
    magnitudes_det = sorted[1]
    inc_det = [i for i in inc_det if i < max_inc]
    omega_det = sorted[2]
    diameter_det = sorted[3]
    
    #source histogram for crossing data 
    plotting.source_hist(sources_crs, f"{orbit_type} {year} {seed} crossings: source histogram", directory, year, orbit_type)
    TLE_mag_crs, frag_mag_crs, rest_mag_crs = sortdata.sort_for_sources(magnitudes_crs, sources_crs)
    TLE_inc_crs, frag_inc_crs, rest_inc_crs = sortdata.sort_for_sources(inc_crs, sources_crs)
    TLE_omega_crs, frag_omega_crs, rest_omega_crs = sortdata.sort_for_sources(omega_crs, sources_crs)
    TLE_diameter_crs, frag_diameter_crs, rest_diameter_crs = sortdata.sort_for_sources(diameter_crs, sources_crs)

    #source histogram for detection data
    plotting.source_hist(sources_det, f"{orbit_type} {year} {seed} detections: source histogram", directory, year, orbit_type)
    TLE_mag_det, frag_mag_det, rest_mag_det = sortdata.sort_for_sources(magnitudes_det, sources_det)
    TLE_inc_det, frag_inc_det, rest_inc_det = sortdata.sort_for_sources(inc_det, sources_det)
    TLE_omega_det, frag_omega_det, rest_omega_det = sortdata.sort_for_sources(omega_det, sources_det)
    TLE_diameter_det, frag_diameter_det, rest_diameter_det = sortdata.sort_for_sources(diameter_det, sources_det)

    #adding rest to the fragments, so this data is in the plots, plots are comparable to observation
    frag_mag_crs = np.hstack([frag_mag_crs, rest_mag_crs])
    frag_mag_det = np.hstack([frag_mag_det, rest_mag_det])
    frag_inc_crs = np.hstack([frag_inc_crs, rest_inc_crs])
    frag_inc_det = np.hstack([frag_inc_det, rest_inc_det])
    frag_omega_crs = np.hstack([frag_omega_crs, rest_omega_crs])
    frag_omega_det = np.hstack([frag_omega_det, rest_omega_det])
    frag_diameter_crs = np.hstack([frag_diameter_crs, rest_diameter_crs])
    frag_diameter_det = np.hstack([frag_diameter_det, rest_diameter_det])

    #magnitude plot for crossings (fragments only)
    plotting.magnitude_plot(frag_mag_crs, year, f"{orbit_type} {year} {seed} crossings: magnitude Histogram", orbit_type, directory)

    #magnitude plot for detections (fragments only)
    plotting.magnitude_plot(frag_mag_det, year, f"{orbit_type} {year} {seed} detections: magnitude Histogram", orbit_type, directory)
    #magnitude plot with no seed in title
    plotting.magnitude_plot(frag_mag_det, year, f"Simulated detections {year} ", orbit_type, directory)
    
    
    #size plot for crossings (fragments only)
    plotting.diameter_plot(frag_diameter_crs, year, f"{orbit_type} {year} {seed} crossings: diameter Histogram", directory, orbit_type)

    #size plot for detections (fragments only)
    plotting.diameter_plot(frag_diameter_det, year, f"{orbit_type} {year} {seed} detections: diameter Histogram",  directory, orbit_type)

    #correlation diameter and magnitude in both det and crs (fragments only)
    plotting.correlation_magnitudes_sizes(frag_mag_crs, frag_mag_det, frag_diameter_crs, frag_diameter_det, year, f"{orbit_type} {year}: correlation diameter vs. magnitude", orbit_type, directory)

    #i Omega plot and ratio (fragments only)
    plotting.i_omega_with_ratio(frag_omega_crs, frag_inc_crs, year, f"{orbit_type} {year} {seed} Inclination vs. RAAN", year, orbit_type, directory, frag_omega_det, frag_inc_det)
    
    #separate plots
    plotting.i_omega_separate(frag_omega_det, frag_inc_det, year, f"{orbit_type} {year} {seed} Inclination vs. RAAN", year, orbit_type, directory, frag_omega_crs, frag_inc_crs)
    
    #print corrected ratio excluding all objects with inclination i > 40°
    calculations.corrected_ratio(frag_inc_det, frag_inc_crs)
    
    return frag_inc_det, frag_inc_crs

def main_unique_all_frags(det1_crs0: bool, all_obj_all_seeds: np.array):
    tolerances = [0, 0.1, np.Inf, 0, np.Inf, np.Inf, np.Inf, np.Inf, 0.001, 0.1, 0.001, 0.1, 0.1, 1, 1, 1, np.Inf, 0.1, np.Inf, np.Inf, 1, 1, np.Inf, np.Inf, np.Inf]
    
    total = 0
    num_unique_objects = 0

    for i in len(0, len(all_obj_all_seeds)):
        objects = all_obj_all_seeds[i]
        total += len(objects)
        unique_objects_list, num_unique_objects = unique_objects.find_unique_objects(objects, tolerances), len(unique_objects.find_unique_objects(objects, tolerances))
        num_unique_objects += num_unique_objects
        
    total = total/ 4
    num_unique_objects = num_unique_objects/4
    ratio = num_unique_objects/total
            
    if det1_crs0:  # If processing detection files
        print(f"Out of {total} detected objects, {num_unique_objects} are unique.")
        print(f"This gives the ratio detected vs. total: {ratio}")
    else:  # If processing crossing files
        print(f"Out of {total} crossing objects, {num_unique_objects} are unique.")
        print(f"This gives the ratio crossing vs. total: {ratio}")

    return unique_objects_list


def main_magnitude_cut(crs_file: np.array, det_file: np.array, year: str, orbit_type: str, seed: str, title: str, directory: str):
    #store crossing data in arrays
    data = crs_file
    sources_crs = data[3]
    magnitudes_crs = data[20]
    inc_crs = data[9]
    omega_crs = data[12]
    diameter_crs = data[1]
    
    #only used for apogee sorting
    sem_major_crs = data[8]
    ecc_crs = data[10]
    
    #apogee sorting
    sorted = sortdata.sort_for_apogee(sem_major_crs, ecc_crs, sources_crs, magnitudes_crs, inc_crs, omega_crs, diameter_crs)
    sources_crs = sorted[0]
    magnitudes_crs = sorted[1]
    inc_crs = sorted[2]
    omega_crs = sorted[3]
    diameter_crs = sorted[4]
    
    #inclination sorting
    max_inc = 22
    sorted = sortdata.sort_for_inclination(inc_crs, max_inc, sources_crs, magnitudes_crs, omega_crs, diameter_crs)
    sources_crs = sorted[0]
    magnitudes_crs = sorted[1]
    inc_crs = [i for i in inc_crs if i < max_inc]
    omega_crs = sorted[2]
    diameter_crs = sorted[3]
    
    #store detected data in arrays
    data = det_file
    sources_det = data[3]
    magnitudes_det = data[20]
    inc_det = data[9]
    omega_det = data[12]
    diameter_det = data[1]
    
    #only used for apogee sorting
    sem_major_det = data[8]
    ecc_det = data[10]

    #apogee sorting
    sorted = sortdata.sort_for_apogee(sem_major_det, ecc_det, sources_det, magnitudes_det, inc_det, omega_det, diameter_det, ecc_det, sem_major_det)
    sources_det = sorted[0]
    magnitudes_det = sorted[1]
    inc_det = sorted[2]
    omega_det = sorted[3]
    diameter_det = sorted[4]
    ecc_det = sorted[5]
    sem_major_det = sorted[6]
    
    #inclination sorting
    max_inc = 22
    sorted = sortdata.sort_for_inclination(inc_det, max_inc, sources_det, magnitudes_det, omega_det, diameter_det, ecc_det, sem_major_det)
    sources_det= sorted[0]
    magnitudes_det = sorted[1]
    inc_det = [i for i in inc_det if i < max_inc]
    omega_det = sorted[2]
    diameter_det = sorted[3]
    ecc_det = sorted[4]
    sem_major_det = sorted[5]
    
    min_mag = 14.5
    sorted = sortdata.sort_for_magnitudes(magnitudes_det, min_mag, sources_det, magnitudes_det, inc_det, omega_det, diameter_det, ecc_det, sem_major_det)
    sources_det = sorted[0]
    magnitudes_det = sorted[1]
    inc_det = sorted[2]
    omega_det = sorted[3]
    diameter_det = sorted[4]
    ecc_det = sorted[5]
    sem_major_det = sorted[6]
    
    #source histogram for crossing data 
    plotting.source_hist(sources_crs, f"{orbit_type} {year} {seed} crossings: source histogram", directory, year, orbit_type)
    TLE_mag_crs, frag_mag_crs, rest_mag_crs = sortdata.sort_for_sources(magnitudes_crs, sources_crs)
    TLE_inc_crs, frag_inc_crs, rest_inc_crs = sortdata.sort_for_sources(inc_crs, sources_crs)
    TLE_omega_crs, frag_omega_crs, rest_omega_crs = sortdata.sort_for_sources(omega_crs, sources_crs)
    TLE_diameter_crs, frag_diameter_crs, rest_diameter_crs = sortdata.sort_for_sources(diameter_crs, sources_crs)

    #source histogram for detection data
    plotting.source_hist(sources_det, f"{orbit_type} {year} {seed} detections: source histogram", directory, year, orbit_type)
    TLE_mag_det, frag_mag_det, rest_mag_det = sortdata.sort_for_sources(magnitudes_det, sources_det)
    TLE_inc_det, frag_inc_det, rest_inc_det = sortdata.sort_for_sources(inc_det, sources_det)
    TLE_omega_det, frag_omega_det, rest_omega_det = sortdata.sort_for_sources(omega_det, sources_det)
    TLE_diameter_det, frag_diameter_det, rest_diameter_det = sortdata.sort_for_sources(diameter_det, sources_det)
    TLE_ecc_det, frag_ecc_det, rest_ecc_det = sortdata.sort_for_sources(ecc_det, sources_det)
    TLE_sem_major_det, frag_sem_major_det, rest_sem_major_det = sortdata.sort_for_sources(sem_major_det, sources_det)

    #adding rest to the fragments, so this data is in the plots, plots are comparable to observation
    frag_mag_crs = np.hstack([frag_mag_crs, rest_mag_crs])
    frag_mag_det = np.hstack([frag_mag_det, rest_mag_det])
    frag_inc_crs = np.hstack([frag_inc_crs, rest_inc_crs])
    frag_inc_det = np.hstack([frag_inc_det, rest_inc_det])
    frag_omega_crs = np.hstack([frag_omega_crs, rest_omega_crs])
    frag_omega_det = np.hstack([frag_omega_det, rest_omega_det])
    frag_diameter_crs = np.hstack([frag_diameter_crs, rest_diameter_crs])
    frag_diameter_det = np.hstack([frag_diameter_det, rest_diameter_det])
    frag_ecc_det = np.hstack([frag_ecc_det, rest_ecc_det])
    frag_sem_maj_det = np.hstack([frag_sem_major_det, rest_sem_major_det])

    #magnitude plot for crossings (fragments only)
    plotting.magnitude_plot(frag_mag_crs, year, f"Simulated crossings {orbit_type} {year} {seed} {title}", orbit_type, directory)

    #magnitude plot for detections (fragments only)
    plotting.magnitude_plot(frag_mag_det, year, f"Simulated detections {orbit_type} {year} {seed} {title}", orbit_type, directory)
    #Magnitude plot for detections, no seed in title
    plotting.magnitude_plot(frag_mag_det, year, f"Simulated detections {year} {title}", orbit_type, directory)

    
    #size plot for crossings (fragments only)
    plotting.diameter_plot(frag_diameter_crs, year, f"Simulated crossings {orbit_type} {year} {seed} {title}", directory, orbit_type)

    #size plot for detections (fragments only)
    plotting.diameter_plot(frag_diameter_det, year, f"Simulated detections {orbit_type} {year} {seed} {title}",  directory, orbit_type)

    #correlation diameter and magnitude in both det and crs (fragments only)
    plotting.correlation_magnitudes_sizes(frag_mag_crs, frag_mag_det, frag_diameter_crs, frag_diameter_det, year, f"{orbit_type} {year}: correlation diameter vs. magnitude {title}", orbit_type, directory)

    #i Omega plot and ratio (fragments only)
    plotting.i_omega_with_ratio(frag_omega_crs, frag_inc_crs, year, f"Simulated crossings vs. simulated detections {year} {seed} {title}", year, orbit_type, directory, frag_omega_det, frag_inc_det)
    #no seed in title
    plotting.i_omega_with_ratio(frag_omega_crs, frag_inc_crs, year, f"Simulated crossings vs. simulated detections {year} {title}", year, orbit_type, directory, frag_omega_det, frag_inc_det)
    
    #separate plots
    plotting.i_omega_separate(frag_omega_det, frag_inc_det, year, f"Simulated detections  {orbit_type} {year} {seed} {title}", year, orbit_type, directory, frag_omega_crs, frag_inc_crs)
    #no seed in title
    plotting.i_omega_separate(frag_omega_det, frag_inc_det, year, f"Simulated detections  {year} {title}", year, orbit_type, directory, frag_omega_crs, frag_inc_crs)
    
    #plot with eccentricity and semi major axis and magnitude colors 
    #plotting.i_omega_with_eccentricity(frag_omega_det, frag_inc_det, frag_ecc_det, f"Simulated detections {year} {seed} {title}", year, directory)
    #plotting.i_omega_with_sem_maj(frag_omega_det, frag_inc_det, frag_sem_maj_det, f"Simulated detections {year} {seed} {title}", year, directory)
    #plotting.i_omega_with_sem_maj(frag_omega_det, frag_inc_det, frag_mag_det, f"Simulated detections {year} {seed} {title}", year, directory)
    #no seed in title 
    plotting.i_omega_with_eccentricity(frag_omega_det, frag_inc_det, frag_ecc_det, f"Simulated detections {year} {title}", year, directory)
    plotting.i_omega_with_sem_maj(frag_omega_det, frag_inc_det, frag_sem_maj_det, f"Simulated detections {year} {title}", year, directory)
    plotting.i_omega_with_sem_maj(frag_omega_det, frag_inc_det, frag_mag_det, f"Simulated detections {year} {title}", year, directory)

    plotting.i_omega_per_size(frag_inc_det, frag_omega_det, frag_diameter_det, year, directory, False)

    #print corrected ratio excluding all objects with inclination i > 22°
    calculations.corrected_ratio(frag_inc_det, frag_inc_crs)
    
    magnitudes_of_8degree_band = []
    for idx, m in enumerate(frag_inc_det): 
        if m > 6 and m < 9: 
            magnitudes_of_8degree_band.append(magnitudes_det[idx])
    mean = np.mean(magnitudes_of_8degree_band)
    median = np.median(magnitudes_of_8degree_band)
    #print(f"Mean of magnitudes of i = 8° band: {mean:.3f}, median: {median:.3f}")
    #print(magnitudes_of_8degree_band)
    
    return frag_inc_det, frag_inc_crs