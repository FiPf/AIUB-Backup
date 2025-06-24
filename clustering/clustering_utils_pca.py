from collections import namedtuple

ClusteringResult = namedtuple("ClusteringResult", ["labels", "cluster_centers", "data"])
ClusterData = namedtuple("ClusterData", ["ecc", "mag_obj", "sem_maj", "diameter", "inc", "raan"])
