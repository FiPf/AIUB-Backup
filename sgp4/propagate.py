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

def modify_eccentricity(tle2):
    """Extracts eccentricity from TLE line 2, adds the zero in front and reconstructs the line"""
    tle_parts = tle2.split()
    if len(tle_parts) < 8:
        raise ValueError(f"Unexpected TLE format: {tle2}")

    # Extract and modify eccentricity (7th field in TLE format)
    eccentricity_str = tle2[27:33]  # Eccentricity is stored as 7 digits with an implied decimal
    eccentricity = float(f"0.{eccentricity_str}") # Convert to float and add the missing zero
    tle2_modified = tle2[:26] + f"{eccentricity:.7f}"[2:9] + tle2[33:]  # Reinsert into TLE

    return tle2_modified

def modify_mean_motion(tle2):
    """Extracts mean motion from TLE line 2, converts to rad/min, and reconstructs the line."""
    if len(tle2) < 63:
        raise ValueError(f"Unexpected TLE format: {tle2}")

    # Extract mean motion (last field in TLE format)
    mean_motion_str = tle2[54:63].strip()  # Ensure correct column range
    print("Mean Motion (revs/day):", mean_motion_str)
    
    mean_motion = float(mean_motion_str)  # Convert to float (revs/day)

    # Convert from revs/day to rad/min
    mean_motion_rad_min = mean_motion * (2 * np.pi / 1440)

    # Format back to TLE standard (keeping the same width)
    mean_motion_formatted = f"{mean_motion_rad_min:11.8f}"  # 11 characters, 8 decimals
    tle2_modified = tle2[:52] + mean_motion_formatted + tle2[63:]

    return tle2_modified

def propagate_tles(input_file, output_file):
    """Propagates TLEs to one year after their epoch and saves results."""
    tles = read_tles(input_file)
    print(tles[0])  # Example: ('TLE line 1', 'TLE line 2')

    with open(output_file, 'w') as f, tqdm(total=len(tles), desc="Propagating TLEs", unit="TLE") as pbar:
        for tle1, tle2 in tles:
            tle2_modified = modify_eccentricity(tle2)  # Correct eccentricity first
            tle2_modified = modify_mean_motion(tle2_modified)  # Then correct mean motion
            print("TLE1", tle1)
            print("TLE2", tle2)
            print("Modified TLE2", tle2_modified)
            
            satellite = Satrec.twoline2rv(tle1, tle2_modified)
            print(f"Epoch: {satellite.jdsatepoch} JD")
            print(f"Mean Motion: {satellite.no_kozai} rad/min")
            print("Converted Back:", (satellite.no_kozai * 1440) / (2 * np.pi))  # Should match revs/day
            print(f"Eccentricity: {satellite.ecco}")
            print(f"Inclination: {np.degrees(satellite.inclo)}°")
            print(f"RAAN: {np.degrees(satellite.nodeo)}°")
            print(f"Argument of Perigee: {np.degrees(satellite.argpo)}°")
            print(f"Mean Anomaly: {np.degrees(satellite.mo)}°")
            print(f"BStar: {satellite.bstar}")

            jd_target = get_propagation_date(tle1)
            jd, fr = divmod(jd_target, 1)
            e, r, v = satellite.sgp4(jd, fr)

            if e == 0:
                orbit_elements = astro.r_v_to_elements_sgp4(np.array(r), np.array(v))

                f.write(f"Position: {r}\n")
                f.write(f"Velocity: {v}\n\n")
                f.write("Orbital Elements:\n")
                f.write(f"Semi-major axis: {orbit_elements['semi_major_axis']} km\n")
                f.write(f"Eccentricity: {orbit_elements['eccentricity']}\n")
                f.write(f"Inclination: {orbit_elements['inclination']}°\n")
                f.write(f"RAAN: {orbit_elements['raan']}°\n")
                f.write(f"Argument of Perigee: {orbit_elements['argument_of_perigee']}°\n")
                f.write(f"True Anomaly: {orbit_elements['true_anomaly']}°\n")
                f.write("-------------------------------------------------\n\n")
            else:
                f.write(f"Error propagating {tle1[:24]}: Error: {e}\n\n")
                dist = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
                f.write(f"{r}\n{v}\n\n")
                f.write(f"{dist}\n")
                f.write("-------------------------------------------------\n")

            pbar.update(1)
