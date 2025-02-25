from sgp4.api import Satrec
from sgp4.api import SGP4_ERRORS
import astro
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

def read_tles(file_path: str):
    """read TLEs from a TLE file and return a list of tuples (line 1 and line 2 of TLE format)

    Args:
        file_path (str): name of the TLE file

    Returns:
        (list): list of tuples containing (line1, line2) or the TLE format
        
    """    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [(lines[i].strip(), lines[i+1].strip()) for i in range(0, len(lines), 2)]

def get_propagation_date(tle_line: str):
    """extract epoch from the TLE, convert to JD and add one year

    Args:
        tle_line (str): Line 1 from TLE, contains the epoch

    Returns:
        (float): epoch from TLE plus one year
    """    
    epoch = float(tle_line[19:28].strip())  # Extract epoch from TLE
    year = int(tle_line[18:20])
    year = 2000 + year if year < 57 else 1900 + year  # Handle 2-digit year format
    mjd_epoch = astro.convert_TCA_to_mjd(np.array([epoch]))[0]  # Convert to MJD
    jd_epoch = astro.mjd_to_jd(mjd_epoch)  # Convert to JD
    return jd_epoch + 365.25  # Add one year in Julian days (1 year = 365.25 days)

def propagate_tles(input_file: str, output_file: str, store_intermediate: bool = False):
    """Propagates TLEs to one year after their epoch and saves results.

    Args:
        input_file (str): File containing the TLEs to be propagated.
        output_file (str): Name of the output file where the results will be stored.
        store_intermediate (bool): If True, store intermediate orbital elements at monthly intervals.
    """    
    tles = read_tles(input_file)
    results = []

    with tqdm(total=len(tles), desc="Propagating TLEs", unit="TLE", mininterval=1.0) as pbar:
        for i, (tle1, tle2) in enumerate(tles):
            satellite = Satrec.twoline2rv(tle1, tle2)
            jd_epoch = satellite.jdsatepoch  # Initial epoch
            jd_target = get_propagation_date(tle1)  # One year later

            if store_intermediate:
                # Generate 12 time steps (one per month)
                time_steps = np.linspace(jd_epoch, jd_target, num=13)  # 12 intervals = 13 points
            else:
                time_steps = [jd_target]  # Only final step

            result = f"TLE {i + 1} Propagation:\n"
            for jd in time_steps:
                jd_int, jd_frac = divmod(jd, 1)
                e, r, v = satellite.sgp4(jd_int, jd_frac)

                if e == 0:
                    orbit_elements = astro.r_v_to_elements_sgp4(np.array(r), np.array(v))
                    dist = np.linalg.norm(r)

                    result += (
                        f"Julian Date: {jd:.6f}\n"
                        f"Position: {r}\nVelocity: {v}\n"
                        f"Distance from center of Earth: {dist} km\n"
                        f"Orbital Elements:\n"
                        f"Semi-major axis: {orbit_elements['semi_major_axis']} km\n"
                        f"Eccentricity: {orbit_elements['eccentricity']}\n"
                        f"Inclination: {orbit_elements['inclination']}°\n"
                        f"RAAN: {orbit_elements['raan']}°\n"
                        f"Argument of Perigee: {orbit_elements['argument_of_perigee']}°\n"
                        f"True Anomaly: {orbit_elements['true_anomaly']}°\n"
                        f"-------------------------------------------------\n\n"
                    )
                else:
                    result += (
                        f"Error propagating {tle1[:24]}: Error: {e}\n"
                        f"{r}\n{v}\n"
                        f"-------------------------------------------------\n"
                    )

            results.append(result)
            pbar.update(1)

    # Write all results at once
    with open(output_file, 'w') as f:
        f.writelines(results)

    print(f"Propagation completed. Results saved to '{output_file}'.")
