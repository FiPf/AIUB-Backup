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

    with open(output_file, 'w') as f, tqdm(total=len(tles), desc="Propagating TLEs", unit="TLE") as pbar:
        for tle1, tle2 in tles:
            satellite = Satrec.twoline2rv(tle1, tle2)
            jd_target = get_propagation_date(tle1)
            jd, fr = divmod(jd_target, 1)
            e, r, v = satellite.sgp4(jd, fr)
            
            if e == 0:
                f.write(f"{tle1}\n{tle2}\n\n")
            else:
                f.write(f"Error propagating {tle1[:24]}: Error: {e}\n\n")
                f.write(f"{tle1}\n{tle2}\n\n")
                f.write("-------------------------------------------------")
            
            pbar.update(1)  # Update progress bar
