from collections import namedtuple

#definition of datatypes used inside the folder "clustering"
ClusteringResult = namedtuple("ClusteringResult", ["labels", "cluster_centers", "data"])
ClusterData = namedtuple("ClusterData", ["inc", "raan", "sem_maj"])
