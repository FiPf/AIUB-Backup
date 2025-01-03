import numpy as np
import os

class SpaceObject:
    """
    Class representing a space object with various attributes.
    
    Attributes:
        ID (str): Identifier for the space object.
        diameter (float): Diameter of the space object.
        factor (float): Some factor related to the space object.
        source (float): Source identifier.
        TCA (float): Time of closest approach.
        TCA_RNG (float): Range at the time of closest approach.
        TCA_ALT (float): Altitude at the time of closest approach.
        TCA_RRT (float): Rate at the time of closest approach.
        sem_major (float): Semi-major axis.
        inc (float): Inclination.
        ecc (float): Eccentricity.
        arg_per (float): Argument of periapsis.
        raan (float): Right ascension of the ascending node.
        true_lat (float): True latitude.
        fov_dwell (float): Field of view dwell time.
        ang_vel (float): Angular velocity.
        pathoffs (float): Path offset.
        albedo (float): Albedo.
        phs_ang (float): Phase angle.
        illumination (float): Illumination.
        mag_obj (float): Magnitude of the object.
        mag_backgr (float): Magnitude of the background.
        max_snr (float): Maximum signal-to-noise ratio.
        RA_LOS (float): Right ascension of the line of sight.
        Des_LOS (float): Declination of the line of sight.
    """
    def __init__(self, ID, diameter, factor, source, TCA, TCA_RNG, TCA_ALT, TCA_RRT, sem_major, inc, ecc, arg_per, raan, true_lat, fov_dwell, ang_vel, pathoffs, albedo, phs_ang, illumination, mag_obj, mag_backgr, max_snr, RA_LOS, Des_LOS):
        self.ID = ID
        self.diameter = diameter
        self.factor = factor
        self.source = source
        self.TCA = TCA
        self.TCA_RNG = TCA_RNG
        self.TCA_ALT = TCA_ALT
        self.TCA_RRT = TCA_RRT
        self.sem_major = sem_major
        self.inc = inc
        self.ecc = ecc
        self.arg_per = arg_per
        self.raan = raan
        self.true_lat = true_lat
        self.fov_dwell = fov_dwell
        self.ang_vel = ang_vel
        self.pathoffs = pathoffs
        self.albedo = albedo
        self.phs_ang = phs_ang
        self.illumination = illumination
        self.mag_obj = mag_obj
        self.mag_backgr = mag_backgr
        self.max_snr = max_snr
        self.RA_LOS = RA_LOS
        self.Des_LOS = Des_LOS

    @classmethod
    def from_list(cls, data):
        """Creates a SpaceObject instance from a list of attributes

        Args:
            data (list): attributes

        Returns:
            SpaceObject
        """
        return cls(*data)

    def to_list(self):
        """
        Convert the SpaceObject attributes to a list.
        
        Returns:
            list: List of attributes.
        """
        return [self.ID, self.diameter, self.factor, self.source, self.TCA, self.TCA_RNG, self.TCA_ALT, self.TCA_RRT, self.sem_major, self.inc, self.ecc, self.arg_per, self.raan, self.true_lat, self.fov_dwell, self.ang_vel, self.pathoffs, self.albedo, self.phs_ang, self.illumination, self.mag_obj, self.mag_backgr, self.max_snr, self.RA_LOS, self.Des_LOS]

    def __eq__(self, other):
        """
        Check if two SpaceObject instances are equal based on their attributes.
        
        Args:
            other (SpaceObject): Another SpaceObject instance.
            
        Returns:
            bool: True if the instances are equal, False otherwise.
        """
        return self.to_list() == other.to_list()

def parse_objects_from_file(*file_paths: str):
    """create SpaceObject objects from a *.det or *.crs file

    Args:
        file_path (str): *.det or *.crs file

    Returns:
        objects (list): list of SpaceObjects instances
    """
    objects = []
    for file_path in file_paths: 
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) != 25: #ensure that there are enough attributes
                    continue
                identifier = parts[0]
                try: #try to convert attributes to float and create a SpaceObject
                    attributes = [float(x) for x in parts[1:]]
                    objects.append(SpaceObject.from_list([identifier] + attributes))
                except ValueError:
                    print(f"Skipping line with non-numeric data: {parts}")
                    continue
    return objects

def objects_equal(obj1, obj2, tolerances):
    """compare two SpaceObject instances based on a tolerance value for each attribute

    Args:
        obj1 (SpaceObject): first instance of SpaceObject
        obj2 (SpaceObject): second instance of SpaceObject
        tolerances (np.array): tolerance for each attribute, value is symmetric around the center

    Returns:
        (bool): true if they are equal within the tolerances, false otherwise
    """
    attrs1 = obj1.to_list()[1:]
    attrs2 = obj2.to_list()[1:]
    return all(abs(a - b) <= tol for a, b, tol in zip(attrs1, attrs2, tolerances[1:]))

def find_unique_objects(objects, tolerances):
    """
    Find unique SpaceObject instances within a list based on tolerances.
    
    Args:
        objects (np.array): List of SpaceObject instances.
        tolerances (np.array): List of tolerance values for each attribute.
        
    Returns:
        list: List of unique SpaceObject instances.
    """
    unique_objects = []
    for obj in objects:
        if not any(objects_equal(obj, u_obj, tolerances) for u_obj in unique_objects):
            unique_objects.append(obj)
    return unique_objects

def process_space_objects(*file_paths, tolerances):
    """
    Process space objects from a file to find unique objects.
    
    Args:
        file_path (str): Path to the file containing space object data.
        tolerances (np.array): List of tolerance values for each attribute.
        
    Returns:
        unique_objects (list): ist of unique objects
        len(unique_objects) (int): number of unique objects
    """
    unique_objects = []
    space_objects = []
    for file_path in file_paths: 
        space_object = parse_objects_from_file(file_path)
        space_objects.append(space_object)
        unique_object = find_unique_objects(space_object, tolerances)
        unique_objects.append(unique_object)
    print(f"Total number of objects: {len(space_objects)}") 
    return unique_objects, len(unique_objects)

def process_space_objects_from_array(objects_array, tolerances):
    """
    Process space objects from a numpy array to find unique objects.
    
    Args:
        objects_array (np.array): Numpy array where each row is a space object with attributes.
        tolerances (np.array): List of tolerance values for each attribute.
        
    Returns:
        unique_objects (list): List of unique SpaceObject instances.
        num_unique_objects (int): Number of unique objects.
    """
    space_objects = [SpaceObject.from_list(obj) for obj in objects_array]
    unique_objects = find_unique_objects(space_objects, tolerances)
    return unique_objects
