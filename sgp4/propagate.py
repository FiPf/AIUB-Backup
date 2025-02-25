from sgp4.api import Satrec
from sgp4.api import SGP4_ERRORS
import astro
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

def read_tles(file_path):
    """Reads TLEs from a text file and returns them as a list of tuples."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [(lines[i].strip(), lines[i+1].strip()) for i in range(0, len(lines), 2)]

def get_propagation_date(tle_line):
    """Extracts epoch from TLE, converts to JD, and adds one year."""
    epoch = float(tle_line[19:28].strip())  # Extract epoch from TLE
    year = int(tle_line[18:20])
    year = 2000 + year if year < 57 else 1900 + year  # Handle 2-digit year format
    mjd_epoch = astro.convert_TCA_to_mjd(np.array([epoch]))[0]  # Convert to MJD
    jd_epoch = astro.mjd_to_jd(mjd_epoch)  # Convert to JD
    return jd_epoch + 1  # Add one year in Julian days

def propagate_tles(input_file, output_file):
    """Propagates TLEs to one year after their epoch and saves results."""
    tles = read_tles(input_file)
    results = []
    
    with tqdm(total=len(tles), desc="Propagating TLEs", unit="TLE", mininterval=1.0) as pbar:
        for i, (tle1, tle2) in enumerate(tles):
            satellite = Satrec.twoline2rv(tle1, tle2)
            jd_target = get_propagation_date(tle1)
            jd, fr = divmod(jd_target, 1)

            e, r, v = satellite.sgp4(jd, fr)

            if e == 0:
                orbit_elements = astro.r_v_to_elements_sgp4(np.array(r), np.array(v))
                dist = np.linalg.norm(r)

                result = (
                    f"Position: {r}\nVelocity: {v}\n\n"
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
                result = (
                    f"Error propagating {tle1[:24]}: Error: {e}\n\n"
                    f"{r}\n{v}\n\n"
                    f"{dist}\n"
                    f"-------------------------------------------------\n"
                )

            results.append(result)
            if i % 100 == 0:
                print(f"Processing {i+1}/{len(tles)} TLEs")

            pbar.update(1)

    # Write all results at once
    with open(output_file, 'w') as f:
        f.writelines(results)