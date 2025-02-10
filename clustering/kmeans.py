import numpy as np
from scipy.spatial.distance import cdist
from cluster_data import ClusteringResult

def k_means(data: np.array, k: int, init: str = 'random', initial_centers: np.array = None, max_iter: int = 300, tol: float = 1e-4):
    """Perform k-means clustering. Runtime: O(n*k*d*i) with number of data points n, number of clusters k, number of 
    dimensions d and number of iterations until convergence i

    Args:
        data (np.array): data to be clustered
        k (int): number of clusters
        init (str, optional): Initailization can either be 'random' or 'guess' provided by the user or 'kmeans++' see function below.
        Defaults to 'random'.
        initial_centers (np.array, optional): If 'guess', then initial_centers must be provided. Defaults to None.
        max_iter (int, optional): Maximum number of iterations. Defaults to 300.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-4.

    Returns:
        ClusteringResult: namedtuple with fields labels, cluster_centers, data.
    """    
    data = np.array(data)
    n, d = data.shape #n number of data points, d number of dimensions in the data

    #First initialization of cluster centers: either with initial guess or randomly
    #Random option: Note that the distortion function is not convex, the algorithm might get stuck in a local minimum. 
    #Rerun the algorithm with various random initializations and take the run with the minimal distortion function. 
    if init == 'guess' and initial_centers is not None:
        centers = initial_centers.copy()
    if init == 'kmeans++':
        centers = kmeans_pp_initialization(data, k)
    else:
        indices = np.random.choice(n, k, replace=False)
        centers = data[indices]

    for iteration in range(max_iter):
        # Assign each point to the closest center
        distances = np.linalg.norm(data[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)  # shape (n, k)
        labels = np.argmin(distances, axis=1)
        
        # Update centers
        new_centers = np.zeros_like(centers)
        for i in range(k):
            if np.sum(labels == i) > 0:
                new_centers[i] = np.mean(data[labels == i], axis=0)
            else:
                # If a cluster loses all its points, reinitialize to a random data point.
                new_centers[i] = data[np.random.choice(n)]
                
        if np.linalg.norm(new_centers - centers) < tol:
            centers = new_centers
            break
        centers = new_centers
    
    return ClusteringResult(labels=labels, cluster_centers=centers, data=data)

def compute_distortion(clustering_result: ClusteringResult) -> float:
    """
    Computes the distortion function J(c, μ) for k-means clustering.
    J(c, μ) = sum(||x(i) - μc(i) ||^2)
    k means clustering is the same as coordinate descent on J, but J is not convex! The algorithm might get stuck in a local minimum. 
    Rerun k means with various random initializations and take the run with the minimal distortion function. 
    
    Parameters:
        clustering_result (ClusteringResult): 
        Named tuple containing:
        - labels: Cluster assignment for each data point
        - cluster_centers: Array of cluster centroids
        - data: Original data points (as a NumPy array)
    
    Returns:
        distortion (float): The total distortion (sum of squared distances to cluster centroids).
    """
    data = np.array([clustering_result.data.inc, 
                     clustering_result.data.raan, 
                     clustering_result.data.ecc]).T  # Shape (m, 3)
    
    labels = np.array(clustering_result.labels)
    centroids = np.array(clustering_result.cluster_centers)
    
    distortion = np.sum(np.linalg.norm(data - centroids[labels], axis=1) ** 2)
    
    return distortion

def kmeans_pp_initialization(data, k):
    """
    Initializes cluster centers using the K-means++ method. https://en.wikipedia.org/wiki/K-means%2B%2B

    Parameters:
        data (np.array): Data points (n x d).
        k (int): Number of clusters.

    Returns:
        np.array: Initialized cluster centers.
    """
    data = np.array(data)
    n, d = data.shape
    centers = np.zeros((k, d))

    # Step 1: Choose the first center randomly
    centers[0] = data[np.random.choice(n)]

    # Step 2: Choose remaining k-1 centers
    for i in range(1, k):
        # Compute the squared distances from the closest center
        distances = np.min(cdist(data, centers[:i]), axis=1)
        probs = distances ** 2  # Probability proportional to D(x)^2
        probs /= np.sum(probs)  # Normalize to get probabilities

        # Select the next center with weighted probability
        centers[i] = data[np.random.choice(n, p=probs)]

    return centers