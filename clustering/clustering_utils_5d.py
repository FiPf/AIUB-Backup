from collections import namedtuple

ClusteringResult = namedtuple("ClusteringResult", ["labels", "cluster_centers", "data"])
ClusterData = namedtuple("ClusterData", ["inc", "raan", "perigee", "ecc", "mag"])
