import numpy as np
from astropy.constants import G, M_earth
from astropy import units as u
from astropy.time import Time
from datetime import datetime, timedelta

#sgp4 stuff
from sgp4.api import Satrec
from astropy import units as u
from poliastro.twobody import Orbit
from poliastro.bodies import Earth

def cartesian_to_keplerian(r: np.array, v: np.array):
    """convert cartesian coordinates and velocity to six Kepler orbital elements

    Args:
        r (np.array): 3d position vector in km
        v (np.array): 3d velocity vector in km/s

    Returns:
        dict: contains the calculated orbital elements

    Note: Summary of TEME (True Equator Mean Equinox) Definition
    -> TEME is an Earth-Centered Inertial (ECI) frame**, meaning it does not rotate with the Earth.  
    -> "True Equator": The plane **perpendicular to the Celestial Ephemeris Pole (CEP)** at a given time, which accounts for Earth's nutation (small periodic oscillations).  
    -> "Mean Equator": A plane perpendicular to an axis that excludes nutation effects but includes precession.  
    -> "Mean Equinox": The intersection of the **Mean Equator and Ecliptic**, projected onto the True Equator. This differs from the classical equinox.  
    -> "Of Date": Indicates that the **precession motion** (a slow shift of CEP over ~25,700 years) is included.  
    Thus, **TEME aligns its X-axis with a projected Mean Equinox but uses a True Equator, making it unique among ECI frames**.
    """    
    r = np.array(r)
    v = np.array(v)

    #precompute absolute vals
    r_mag = np.linalg.norm(r)  # km
    v_mag = np.linalg.norm(v)  # km/s

    # Standard gravitational parameter for Earth 
    mu = (G * M_earth).to(u.km**3 / u.s**2).value #(km^3/s^2)

    # Compute specific angular momentum vector
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Compute inclination 
    inclination = np.degrees(np.arccos(h[2] / h_mag))

    K = np.array([0, 0, 1])  
    n = np.cross(K, h) 
    n_mag = np.linalg.norm(n)

    # Compute eccentricity vector
    e_vec = (np.cross(v, h) / mu) - (r / r_mag)
    eccentricity = np.linalg.norm(e_vec)

    # Compute RAAN
    if n_mag != 0:
        RAAN = np.degrees(np.arccos(n[0] / n_mag))
        if n[1] < 0:
            RAAN = 360 - RAAN
    else:
        RAAN = 0

    # Compute argument of perigee 
    if n_mag != 0 and eccentricity > 0:
        omega = np.degrees(np.arccos(np.dot(n, e_vec) / (n_mag * eccentricity)))
        if e_vec[2] < 0:
            omega = 360 - omega
    else:
        omega = 0

    # Compute true anomaly
    if eccentricity > 0:
        true_anomaly = np.degrees(np.arccos(np.dot(e_vec, r) / (eccentricity * r_mag)))
        if np.dot(r, v) < 0:
            true_anomaly = 360 - true_anomaly
    else:
        true_anomaly = 0

    # Compute semi-major axis
    energy = (v_mag**2 / 2) - (mu / r_mag)
    if abs(energy) > 1e-10:
        semi_major_axis = -mu / (2 * energy)
    else:
        semi_major_axis = np.inf  # If energy is zero, it's a parabolic trajectory, should not happen hopefully

    return {
        "Semi-Major Axis (km)": semi_major_axis,
        "Eccentricity": eccentricity,
        "Inclination (deg)": inclination,
        "RAAN (deg)": RAAN,
        "Argument of Perigee (deg)": omega,
        "True Anomaly (deg)": true_anomaly
    }

def orbital_to_cartesian(a: float, e: float, i: float, Omega: float, omega: float, nu: float, mu=398600.4418):  # mu in km^3/s^2 for Earth¨
    """transforms orbital elements to cartesian coordinates

    Args:
        a (float): semi major axis
        e (float): eccentricity
        i (float): inclination
        Omega (float): Right ascenciosion of ascending node
        omega (float): argument of perigee
        nu (float): true anomaly
        mu (float, optional): Gravitation value. Defaults to 398600.4418. (Earth)

    Returns: 
        position_inertial (np.array): 3d position vector
        velocity_inertial (np.array): 3d velocity vector
        
    """    
    # Step 1: Calculate distance r
    r = a * (1 - e**2) / (1 + e * np.cos(np.radians(nu)))
    
    # Step 2: Position in orbital plane
    x_prime = r * np.cos(np.radians(nu))
    y_prime = r * np.sin(np.radians(nu))
    z_prime = 0  # Always 0 in the orbital plane

    # Step 3: Velocity in orbital plane
    p = a * (1 - e**2)  # Semi-latus rectum
    vx_prime = -np.sqrt(mu / p) * np.sin(np.radians(nu))
    vy_prime = np.sqrt(mu / p) * (e + np.cos(np.radians(nu)))
    vz_prime = 0  # Always 0 in the orbital plane

    # Step 4: Transformation matrices for inclination, RAAN, and argument of periapsis
    R3_Omega = np.array([[np.cos(np.radians(-Omega)), -np.sin(np.radians(-Omega)), 0],
                         [np.sin(np.radians(-Omega)),  np.cos(np.radians(-Omega)), 0],
                         [0,                          0,                         1]])
    
    R1_i = np.array([[1,  0,                        0],
                     [0,  np.cos(np.radians(-i)),  -np.sin(np.radians(-i))],
                     [0,  np.sin(np.radians(-i)),   np.cos(np.radians(-i))]])
    
    R3_omega = np.array([[np.cos(np.radians(-omega)), -np.sin(np.radians(-omega)), 0],
                         [np.sin(np.radians(-omega)),  np.cos(np.radians(-omega)), 0],
                         [0,                          0,                         1]])
    
    # Combined rotation matrix
    rotation_matrix = R3_Omega @ R1_i @ R3_omega

    # Step 5: Transform position and velocity to inertial frame
    position_orbital = np.array([x_prime, y_prime, z_prime])
    velocity_orbital = np.array([vx_prime, vy_prime, vz_prime])

    position_inertial = rotation_matrix @ position_orbital
    velocity_inertial = rotation_matrix @ velocity_orbital

    return position_inertial, velocity_inertial

def convert_TCA_to_mjd(dates: np.array):
    """Converts an array of dates in YYDDD.ddd format to MJD (Modified Julian Date). 
    
    Args:
        dates (np.array): List of dates in the YYDDD.ddd format. 
            - For one-digit years (before 2010), example: 5068.2400303 (meaning 1950).
            - For two-digit years (after 2000), example: 21072.195442 (meaning 2021).
    
    Returns: 
        np.array: Array of MJD values.
    """
    mjd_array = []
    
    for date in dates:
        # Convert to string and extract year & day-of-year
        year_day_str = str(date)
        year_part = int(year_day_str[:2])  # First two characters (YY)
        day_of_year = float(year_day_str[2:])  # Extract DDD.ddd part
        
        # Determine full year using NORAD convention
        if year_part >= 57:  # 1957 or later means it's in the 1900s
            year = 1900 + year_part
        else:  # 2056 or earlier means it's in the 2000s
            year = 2000 + year_part
        
        # Convert fractional day to full datetime
        base_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        
        # Compute MJD (Modified Julian Date)
        mjd = (base_date - datetime(1858, 11, 17)).total_seconds() / 86400.0
        
        # Append to array
        mjd_array.append(mjd)

    return np.array(mjd_array)

def mjd_to_date(mjd):
    """
    Convert Modified Julian Date (MJD) to a standard date format.
    
    Parameters:
    mjd (float): Modified Julian Date to be converted.
    
    Returns:
    str: Date in the format YYYY-MM-DD HH:MM:SS
    """
    # MJD reference date: 17 November 1858
    mjd_ref_date = datetime(1858, 11, 17)
    # Convert MJD to date
    date = mjd_ref_date + timedelta(days=mjd)
    return date.strftime('%Y-%m-%d %H:%M:%S')

def date_to_mjd_manual(date_string):
    """
    Convert a date and time to Modified Julian Date (MJD) manually without Astropy.
    
    Parameters:
        date_string (str): A single string containing date and time in the format
                           'YYYY MM DD HH MM SS.SSS' (e.g., '2005 1 4 20 48 58.000').
    
    Returns:
        float: The corresponding Modified Julian Date (MJD).
    """
    # Parse the date string
    year, month, day, hour, minute, second = map(float, date_string.split())

    # Adjust months and years if the month is January or February
    if month <= 2:
        year -= 1
        month += 12

    # Calculate the Julian Date (JD)
    A = int(year // 100)
    B = 2 - A + (A // 4)  # Gregorian calendar adjustment
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    JD += (hour / 24.0) + (minute / 1440.0) + (second / 86400.0)

    # Convert JD to MJD
    MJD = JD - 2400000.5
    return MJD

def jd_to_gregorian(jd, fr=0.0):
    """
    Converts a Julian Date (JD) and fractional day (fr) to a Gregorian calendar date (UTC).
    
    Args:
        jd (float): Julian Date (whole part).
        fr (float, optional): Fractional day (default is 0.0).
    
    Returns:
        str: Gregorian date in 'YYYY-MM-DD HH:MM:SS' format (UTC).
    """
    # Convert Julian Date to ISO format using astropy
    date = Time(jd + fr, format='jd').utc.iso
    
    return date  # Returns as 'YYYY-MM-DD HH:MM:SS.sss'

def mjd_to_jd(mjd: float) -> float:
    """
    Convert Modified Julian Date (MJD) to Julian Date (JD).
    
    JD = MJD + 2400000.5
    
    :param mjd: Modified Julian Date
    :return: Julian Date
    """
    return mjd + 2400000.5

def r_v_to_elements_sgp4(r, v):
    """
    Converts position and velocity vectors in Earth-centered inertial (ECI) coordinates to classical orbital elements.

    Parameters:
        r (list or array): Position vector in km [x, y, z]
        v (list or array): Velocity vector in km/s [vx, vy, vz]

    Returns:
        dict: Dictionary containing orbital elements:
            - semi_major_axis [km]
            - eccentricity
            - inclination [deg]
            - raan [deg]
            - argument_of_perigee [deg]
            - true_anomaly [deg]
    """
    # Convert r and v to astropy units
    r = r * u.km
    v = v * u.km / u.s

    # Create orbit from r and v
    orbit = Orbit.from_vectors(Earth, r, v)

    return {
        "semi_major_axis": orbit.a.to(u.km).value,
        "eccentricity": orbit.ecc.value,
        "inclination": orbit.inc.to(u.deg).value,
        "raan": orbit.raan.to(u.deg).value,
        "argument_of_perigee": orbit.argp.to(u.deg).value,
        "true_anomaly": orbit.nu.to(u.deg).value
    }

def elements_to_r_v_sgp4(orbital_elements: np.array):
    """convert from orbital elements to 3d position and velocity in Earth-centered inertial (ECI) coordinates in km and km/s

    Args:
        orbital_elements (np.array): orbital elements in the following order: 
        orbital_elements (np.array): resulting orbital elements in the following order: 
        -semi major axis [km]
        -eccentricity 
        -inclination [radians]
        -raan [radians]
        -argument of perigee [radians]
        -true anomaly [radians]

    Returns:
        r (np.array): 3d position in Earth-centered inertial (ECI) coordinates in km
        v (np.array): 3d velicity in Earth-centered inertial (ECI) coordinates in km/s
    """     
    a, ecc, inc, raan, argp, nu = orbital_elements

    # Convert to astropy units
    a = a * u.km
    inc = inc * u.deg
    raan = raan * u.deg
    argp = argp * u.deg
    nu = nu * u.deg

    orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
    r, v = orbit.rv()
    return r, v