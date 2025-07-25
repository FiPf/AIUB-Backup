from collections import namedtuple

ClusteringResult = namedtuple("ClusteringResult", ["labels", "cluster_centers", "data"])
ClusterData = namedtuple("ClusterData", ["ecc", "sem_maj", "inc", "raan", "perigee", "true_lat", "mean_motion", "mag_obj", "diameter","cospar_id"]) 
#9 dimensions: orbital elements, mean motion, magnitude and diameter, plus cospar_id
#same as in the file: clustering_utils_pca_08072025.py, except I added the cospar_id, PAY ATTENTION THAT YOU DON'T INCLUDE COSPAR ID IN CLUSTERING