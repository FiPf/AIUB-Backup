from collections import namedtuple

ClusteringResult = namedtuple("ClusteringResult", ["labels", "cluster_centers", "data"])
ClusterData = namedtuple("ClusterData", ["ecc", "sem_maj", "inc", "raan", "perigee", "true_lat", "mean_motion", "mag_obj", "diameter"]) 
#9 dimensions: orbital elements, mean motion, magnitude and diameter
