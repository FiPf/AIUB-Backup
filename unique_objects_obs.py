import numpy as np
import os

class ObsSpaceObject:
    """
    Class representing an observed space object with different attributes.
    
    Attributes:
        Campaign (str): The campaign name.
        Name (str): Name of the space object.
        Osc_epoch (float): Osculating epoch.
        Date (str): Date.
        Time (str): Time.
        arcl (float): Arc length in minutes.
        num_obs (float): Number of observations.
        mag_c (float): Magnitude C.
        mag_u (float): Magnitude U.
        a_km (float): Semi-major axis in kilometers.
        e (float): Eccentricity.
        i_dg (float): Inclination in degrees.
        raan_dg (float): Right ascension of the ascending node in degrees.
        w_dg (float): Argument of periapsis in degrees.
        s_dg (float): S in degrees.
        dn_dg_d (float): DN in degrees per day.
        da_km (float): DA in kilometers.
        de (float): DE.
        di_dg (float): DI in degrees.
        draan_dg (float): DRAAN in degrees.
        dw_dg (float): DW in degrees.
        ds_dg (float): DS in degrees.
        ra_h (float): Right ascension in hours.
        decl_dg (float): Declination in degrees.
    """
    def __init__(self, Campaign, Name, Osc_epoch, Date, Time, arcl, num_obs, mag_c, mag_u, a_km, e, i_dg, raan_dg, w_dg, s_dg, dn_dg_d, da_km, de, di_dg, draan_dg, dw_dg, ds_dg, ra_h, decl_dg):
        self.Campaign = Campaign
        self.Name = Name
        self.Osc_epoch = float(Osc_epoch)
        self.Date = Date
        self.Time = Time
        self.arcl = float(arcl)
        self.num_obs = float(num_obs)
        self.mag_c = float(mag_c)
        self.mag_u = float(mag_u)
        self.a_km = float(a_km)
        self.e = float(e)
        self.i_dg = float(i_dg)
        self.raan_dg = float(raan_dg)
        self.w_dg = float(w_dg)
        self.s_dg = float(s_dg)
        self.dn_dg_d = float(dn_dg_d)
        self.da_km = float(da_km)
        self.de = float(de)
        self.di_dg = float(di_dg)
        self.draan_dg = float(draan_dg)
        self.dw_dg = float(dw_dg)
        self.ds_dg = float(ds_dg)
        self.ra_h = float(ra_h)
        self.decl_dg = float(decl_dg)

    @classmethod
    def from_list(cls, data):
        """Creates an ObsSpaceObject instance from a list of attributes."""
        return cls(*data)

    def to_list(self):
        """Convert the ObsSpaceObject attributes to a list."""
        return [
            self.Campaign, self.Name, self.Osc_epoch, self.Date, self.Time,
            self.arcl, self.num_obs, self.mag_c, self.mag_u, self.a_km, self.e, 
            self.i_dg, self.raan_dg, self.w_dg, self.s_dg, self.dn_dg_d, 
            self.da_km, self.de, self.di_dg, self.draan_dg, self.dw_dg, 
            self.ds_dg, self.ra_h, self.decl_dg
        ]

    def to_numeric_list(self):
        """Convert only numeric attributes of the ObsSpaceObject to a list."""
        return [
            self.Osc_epoch, self.arcl, self.num_obs, self.mag_c, self.mag_u,
            self.a_km, self.e, self.i_dg, self.raan_dg, self.w_dg, self.s_dg, 
            self.dn_dg_d, self.da_km, self.de, self.di_dg, self.draan_dg,
            self.dw_dg, self.ds_dg, self.ra_h, self.decl_dg
        ]

    def __eq__(self, other):
        """Check if two ObsSpaceObject instances are equal based on their attributes."""
        return self.to_list() == other.to_list()


def obs_objects_equal(obj1, obj2, tolerances):
    """Compare two ObsSpaceObject instances based on tolerance values for numeric attributes."""
    attrs1 = obj1.to_numeric_list()
    attrs2 = obj2.to_numeric_list()
    return all(abs(a - b) <= tol for a, b, tol in zip(attrs1, attrs2, tolerances))


def find_unique_obs_objects(objects, tolerances):
    """
    Find unique ObsSpaceObject instances within a list based on tolerances.
    
    Args:
        objects (list): List of ObsSpaceObject instances.
        tolerances (list): List of tolerance values for each attribute.
        
    Returns:
        list: List of unique ObsSpaceObject instances.
    """
    unique_objects = []
    for obj in objects:
        if not any(obs_objects_equal(obj, u_obj, tolerances) for u_obj in unique_objects):
            unique_objects.append(obj)
    return unique_objects

def process_obs_space_objects_from_array(objects_array, tolerances):
    """
    Process observed space objects from a numpy array to find unique objects.
    
    Args:
        objects_array (np.array): Numpy array where each row is a space object with attributes.
        tolerances (list): List of tolerance values for each attribute.
        
    Returns:
        unique_objects (list): List of unique ObsSpaceObject instances.
        num_unique_objects (int): Number of unique objects.
    """
    # Convert each row in the numpy array to an ObsSpaceObject instance
    obs_objects = [ObsSpaceObject.from_list(obj) for obj in objects_array]
    print(len(obs_objects))
    # Find unique ObsSpaceObject instances
    unique_objects = find_unique_obs_objects(obs_objects, tolerances)
    print(len(unique_objects))
    # Return the list of unique objects and their count
    return unique_objects
