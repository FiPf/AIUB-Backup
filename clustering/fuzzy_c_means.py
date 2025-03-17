# fuzzy_c_means.py
import numpy as np
from clustering_utils import ClusteringResult
import skfda
#from skfda.ml.clustering import FuzzyCMeans

def fuzzy_c_means(data: np.array, k: int , m: int =2, max_iter: int =100, tol: float =1e-4):
    """perform fuzzy c means clustering. Runtime: O(i*n*c^2*d) with number of iterations i, number of data points n, 
    number of clusters c, number of dimensions d

    Args:
        data (np.array): data of shape (n, d)
        k (int): number of clusters
        m (int, optional): Fuzziness, m > 1. Defaults to 2.
        max_iter (int, optional): Maximal iterations. Safety stop in case it does not converge. Defaults to 100.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-4.

    Returns:
        ClusteringResult: named tuple with fields labels, cluster_centers, data.

    """     
    n, d = data.shape
    # Initialize membership matrix U with random values (each row sums to 1).
    U = np.random.rand(n, k)
    U = U / np.sum(U, axis=1, keepdims=True)
    
    for iteration in range(max_iter):
        # Compute cluster centers.
        centers = np.zeros((k, d))
        for i in range(k):
            numerator = np.sum((U[:, i]**m).reshape(-1, 1) * data, axis=0)
            denominator = np.sum(U[:, i]**m)
            centers[i] = numerator / denominator
        
        # Update membership matrix U.
        U_new = np.zeros_like(U)
        for j in range(n):
            for i in range(k):
                # Avoid division by zero by adding a small number.
                dist_ij = np.linalg.norm(data[j] - centers[i]) + 1e-10
                denom_sum = 0
                for l in range(k):
                    dist_lj = np.linalg.norm(data[j] - centers[l]) + 1e-10
                    ratio = dist_ij / dist_lj
                    denom_sum += ratio**(2 / (m - 1))
                U_new[j, i] = 1 / denom_sum
        
        # Check for convergence.
        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new
    
    # After convergence, assign hard clusters based on maximum membership.
    labels = np.argmax(U, axis=1)
    return ClusteringResult(labels=labels, cluster_centers=centers, data=data)

def fuzzy_c_means_fda(data: np.array, k: int, m: float = 2.0, max_iter: int = 300, tol: float = 1e-4):
    """
    Perform Fuzzy C-Means clustering using skfda's implementation.

    Args:
        data (np.array): Data points to cluster.
        k (int): Number of clusters.
        m (float, optional): Fuzziness parameter. Defaults to 2.0.
        max_iter (int, optional): Maximum number of iterations. Defaults to 300.
        tol (float, optional): Tolerance for stopping criterion. Defaults to 1e-4.

    Returns:
        ClusteringResult: Named tuple with fields labels, cluster_centers, and data.
    """
    # Initialize the Fuzzy C-Means model
    fcm = FuzzyCMeans(n_clusters=k, m=m, max_iter=max_iter, tol=tol)
    
    # Fit to the data
    fcm.fit(data)
    
    # Get the cluster assignments (hard clustering labels)
    labels = np.argmax(fcm.memberships_, axis=1)
    
    # Extract the cluster centers
    cluster_centers = fcm.cluster_centers_
    
    return ClusteringResult(labels=labels, cluster_centers=cluster_centers, data=data)