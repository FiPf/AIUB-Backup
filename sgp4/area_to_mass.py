import numpy as np
from astropy.time import Time
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath("..")) 
import getdata

from getdata import PopulationType, data_returner
from collections import namedtuple
import os

#uses the formula from Wikipedia
def compute_b_drag(diameter: np.array, semi_major: np.array, sources: np.array):
    """compute the b star drag term, which is used in the TLE for the orbit propagation. 
    Use area-to-mass ratio from PROOF/ MASTER model and use drag_coeff = 2.2 (standard default value). 
    The density is taken from https://en.wikipedia.org/wiki/BSTAR. 

    Args:
        diameter (np.array): object diameter
        semi_major (np.array): semi major axis of the object
        sources (np.array): sources of the object

    Returns:
        b_star_drag_values (np.array): calculated bstar drag term, can be used in the TLE
    """
    b_star_drag_vals = []
    b_vals = [] 
    area_to_mass_vals = []
    for d, s in zip(diameter, sources): 
        a_m = compute_am(d, s) # two possible versions, see below
        area_to_mass_vals.append(a_m)

    drag_coeff = 2.2
    density_const = 0.15696615
    R_earth = 6378

    for ele, a in zip(area_to_mass_vals, semi_major): 
        rho = density_const * (R_earth / a)
        b_star_drag = drag_coeff*ele*rho*0.5
        b_star_drag_vals.append(b_star_drag)
        b = drag_coeff*ele
        b_vals.append(b)

    return b_star_drag_vals, b_vals

#version 1: using the method described in the MASTER final report
def compute_am(d: float, source: int):
    """compute the area-to-mass ratiom A/m for an object of size d. The method for this function is
    described in the MASTER final report! 

    Args:
        d (float): diameter of the object
        source (int): source of the object (PROOF number 1- 6)

    Returns:
        area_to_mass (float): A/m in m^2/kg
    """    
    object_type = ""

    if source == 1: 
        object_type == 'spacecraft'
    if source == 4: 
        object_type == 'rocket'
    else: 
        object_type == 'None' #small objects only, all sources excpet 1 (=fragments) and 4 (=TLE)

    log_d = np.log10(d)
    small_particle_limit = 1.7 #default value in case that object_type is None
    r_thresh = 0 #default value in case that object_type is None
    
    # Define bridging function threshold only 
    if object_type != 'None': 
        if object_type == 'rocket':
            r_thresh = 10 ** (log_d + 1.76)
            small_particle_limit = 1.7  # cm
        elif object_type == 'spacecraft':
            r_thresh = 10 ** (log_d + 1.05)
            small_particle_limit = 8  # cm
    
    # Generate random r for decision making, draw from a uniform distribution
    r = np.random.uniform(0, 1)
    
    # whether to use small or large particle distribution
    if d <= small_particle_limit / 100:  # Convert cm to meters
        alpha, mu1, sigma1 = 1, -0.3, 0.2 if log_d <= -3.5 else 0.2 + 0.1333 * (log_d + 3.5)
        chi = np.random.normal(mu1, sigma1)
    elif d >= 0.11:
        # Large particle model (bi-modal normal distribution)
        delta = log_d
        if object_type == 'rocket':
            if delta <= -1.4:
                alpha = 1
            elif -1.4 < delta < 0:
                alpha = 1 - 0.3571 * (delta + 1.4)
            else:
                alpha = 0.5
            
            mu1 = -0.45 if delta <= -0.5 else -0.45 - 0.9 * (delta + 0.5)
            mu2, sigma1, sigma2 = -0.9, 0.55, 0.28 if delta <= -1 else 0.28 - 0.1636 * (delta + 1)
            sigma2 = max(0.1, sigma2)
        else:  # Spacecraft
            if delta <= -1.95:
                alpha = 0
            elif -1.95 < delta < 0.55:
                alpha = 0.3 + 0.4 * (delta + 1.2)
            else:
                alpha = 1
            
            mu1 = -0.6 if delta <= -1.1 else -0.6 - 0.318 * (delta + 1.1)
            mu2 = -1.2 if delta <= -0.7 else -1.2 - 1.333 * (delta + 0.7)
            mu2 = max(-2, mu2)
            sigma1 = 0.1 if delta <= -1.3 else 0.1 + 0.2 * (delta + 1.3)
            sigma1 = min(0.3, sigma1)
            sigma2 = 0.5 if delta <= -0.5 else 0.5 - (delta + 0.5)
            sigma2 = max(0.3, sigma2)
        
        # Sample from bi-modal distribution
        if np.random.uniform(0, 1) < alpha:
            chi = np.random.normal(mu1, sigma1)
        else:
            chi = np.random.normal(mu2, sigma2)
    else:
        # Bridging function
        if r > r_thresh:
            return compute_am(0.11, source)
        else:
            return compute_am(small_particle_limit / 100, source)
    
    area_to_mass = 10**chi
    return area_to_mass

#version 2: some power law, parameters are just an estimation
def compute_am(d: float, source: int) -> float:
    """
    Compute the area-to-mass ratio A/m for an object of size d.
    The method follows the MASTER final report.
    
    Args:
        d (float): Diameter of the object (m)
        source (int): Source category (1 = spacecraft, 4 = rocket, else = fragments/MLI)
    
    Returns:
        float: A/m in m²/kg
    """
    if source == 1:  # Spacecraft
        return 0.1 * d**-0.5
    elif source == 4:  # Rocket bodies
        return 0.05 * d**-0.4
    else:  # Fragments and MLI
        return 5.0 * d**-0.8