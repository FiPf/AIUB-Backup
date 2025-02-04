import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

#helper function used for all the plots
def save_unique_plot(file_path: str, directory: str) -> str:
    """
    Helper function to ensure a unique filename in the given directory.
    If the filename already exists, append _1, _2, etc., to make it unique.
    Returns the new filename.
    """
    base_name = os.path.basename(file_path) 
    base_path, extension = os.path.splitext(base_name)  # Split into base name and extension, extension e.g. .png
    new_file_path = os.path.join(directory, base_name)
    
    count = 1
    while os.path.exists(new_file_path):  # Check if the file already exists
        new_file_path = os.path.join(directory, f"{base_path}_{count}{extension}") 
        count += 1 
    
    return new_file_path