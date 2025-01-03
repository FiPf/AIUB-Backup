import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

def count_unique_objects_from_id(*filenames: str):
    """looks at the id of the objects in the *.det or *.crs files and removes duplicates
    Args: 
        filenames (strs): filenames to search for unique objects 

    Returns:
        len(unique_sorted_lines) (int): length of uniques
    """
    all_lines = []
    for filename in filenames:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file if '#' not in line]
            all_lines.extend(lines)
    
    print(f"Total number of objects: {len(all_lines)}")
    # Extract unique lines based on the first column (number)
    unique_lines_dict = {}
    for line in all_lines:
        key = line.split()[0]  # Extracting the first column
        if key not in unique_lines_dict:
            unique_lines_dict[key] = line
    
    # Sort the unique lines based on the first column (number)
    unique_sorted_lines = sorted(unique_lines_dict.values())
    
    """for filename in filenames:
        new_filename = 'unique_' + filename
        with open(new_filename, 'w') as new_file:
            for line in unique_sorted_lines:
                new_file.write(line + '\n')"""
    
    return len(unique_sorted_lines)

def count_unique_objects_from_array(*arrays):
    """counts unique objects from objects stored in an array

    Args: 
        arrays (np.arrays): arrays to count unique objects in
    
    Returns:
        unique_counts (np.array): contains the number of unique objects for each array
    """
    unique_counts = []
    
    for array in arrays:
        unique_objects_set = set(array)
        unique_counts.append(len(unique_objects_set))
                
    return unique_counts