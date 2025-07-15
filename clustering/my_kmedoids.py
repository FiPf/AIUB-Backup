import numpy as np
from collections.abc import Callable
from clustering_utils import ClusteringResult
from scipy.spatial.distance import cdist
#from sklearn_extra.cluster import KMedoids
#from kmedoids import KMedoids as KMedoidsLib

#https://arxiv.org/abs/1810.05691 algorithms are based on this paper!

#This once worked, incredibly inefficient!!! Now it is broken :/

def total_cost(data: np.array, medoid_indices: np.array):
    """Compute the total cost given medoid indices for the data.

    Args:
        data (np.array): Dataset
        medoid_indices (np.array): Indices of medoids

    Returns:
        cost (float): Total clustering cost
    """
    cost = 0
    n = data.shape[0]
    medoids = data[medoid_indices]  # Convert indices to actual data points
    for i in range(n):
        cost += min(np.linalg.norm(data[i] - m) for m in medoids)
    return cost

def pam_build(data: np.array , k: int):
    """First part of PAM algorithm. BUILD phase: find initial medoids. Runtime: O(n^2*d) with the number of data points n
    and the number of dimensions in the data d.

    Args:
        data (np.array): data of shape (n, d)
        k (int): number of clusters/medoids

    Returns:
        medoids (list): list of selected medoid indices.
        cost (float): total cost after the build phase.
    """    
    n = data.shape[0]
    medoids = []
    # Choose the first medoid: the point with minimal total distance to all others.
    best_total = np.inf
    best_idx = None
    for j in range(n):
        cost_j = sum(np.linalg.norm(data[j] - data[i]) for i in range(n) if i != j)
        if cost_j < best_total:
            best_total = cost_j
            best_idx = j
    medoids.append(best_idx)
    
    # Iteratively select the remaining medoids.
    while len(medoids) < k:
        best_delta = np.inf
        best_candidate = None
        for candidate in range(n):
            if candidate in medoids:
                continue
            delta = 0
            for i in range(n):
                current_distance = min(np.linalg.norm(data[i] - data[m]) for m in medoids)
                candidate_distance = np.linalg.norm(data[i] - data[candidate])
                diff = candidate_distance - current_distance
                if diff < 0:
                    delta += diff
            if delta < best_delta:
                best_delta = delta
                best_candidate = candidate
        medoids.append(best_candidate)
    cost = total_cost(data, medoids)
    print(np.array(medoids).shape)
    return medoids, cost

def fastpam_lab_build(data, k):
    """
    FastPAM LAB: Linear Approximate BUILD initialization for K-medoids.

    Parameters:
        data (np.array): Data points (n x d).
        k (int): Number of medoids.

    Returns:
        tuple: (total cost TD, list of medoid indices)
    """
    n = data.shape[0]
    subsample_size = int(10 + np.sqrt(n))  # Subsample size

    # Step 1: Select the first medoid
    TD, m1 = float('inf'), None
    S = np.random.choice(n, subsample_size, replace=True)  # Random subsample

    for j in S:
        TDj = np.sum([np.linalg.norm(data[o] - data[j]) for o in S if o != j])  # Sum of distances
        if TDj < TD:
            TD, m1 = TDj, j  # Store best first medoid

    medoids = [m1]

    # Step 2: Select the remaining k-1 medoids
    for i in range(1, k):
        best_TD_reduction, best_x = float('inf'), None
        S = np.random.choice([idx for idx in range(n) if idx not in medoids], subsample_size, replace=True)

        for j in S:
            TD_reduction = 0
            for o in S:
                if o == j:
                    continue
                current_distance = min(np.linalg.norm(data[o] - data[m]) for m in medoids)
                new_distance = np.linalg.norm(data[o] - data[j])
                delta = new_distance - current_distance
                if delta < 0:
                    TD_reduction += delta

            if TD_reduction < best_TD_reduction:
                best_TD_reduction, best_x = TD_reduction, j

        medoids.append(best_x)

    # Assign each point to the nearest medoid
    distances = cdist(data, data[medoids])
    labels = np.argmin(distances, axis=1)
    total_cost = np.sum(np.min(distances, axis=1))  # Compute total distortion

    return medoids, total_cost


def compute_swap_delta(data: np.array, medoids: np.array, m: int, candidate: int):
    """Compute the change in total cost if medoid 'm' is swapped with a candidate.

    Args:
        data (np.array): data
        medoids (np.array): medoids at the current step
        m (int): medoid index to be swapped out
        candidate (int): the candidate non-medoid index to be swapped in

    Returns:
        delta (float): Change in the total cost. Negative if the swap improves the cost. 
    """     
    n = data.shape[0]
    new_medoids = [candidate if x == m else x for x in medoids]
    delta = 0
    for i in range(n):
        old_distance = min(np.linalg.norm(data[i] - data[x]) for x in medoids)
        new_distance = min(np.linalg.norm(data[i] - data[x]) for x in new_medoids)
        delta += (new_distance - old_distance)
    return delta

def pam_swap(data: np.array, medoids: np.array):
    """Second part of PAM algorithm. SWAP phase: iteratively improve medoids by swapping. Runtime: O(k(n-k)^2) with n number of 
    data points and number of medoids k

    Args:
        data (np.array): data of shape (n, d)
        medoids (list): current medoid indices

    Returns:
        new_medoids (list): updated medoid indices after performing the best swap.
        new_cost (float): total cost after the best swap. 
        improved (bool): whether a swap was made
    """    
    n = data.shape[0]
    current_cost = total_cost(data, medoids)
    best_delta = 0
    best_swap = None
    print(medoids, "medoids")
    medoids = list(medoids)
    for m in medoids:
        for candidate in range(n):
            if candidate in medoids:
                continue
            delta = compute_swap_delta(data, medoids, m, candidate)
            if delta < best_delta:
                best_delta = delta
                best_swap = (m, candidate)
    if best_swap is not None:
        m, candidate = best_swap
        new_medoids = [candidate if x == m else x for x in medoids]
        new_cost = current_cost + best_delta
        return new_medoids, new_cost, True
    else:
        return medoids, current_cost, False


def fastpam1_swap(data: np.array, k: int, max_iter: int = 300, tol: float = 1e-4):
    n = data.shape[0]

    # Step 1: Initialize distinct medoids randomly
    medoid_indices = np.random.choice(n, k, replace=False)
    medoids = data[medoid_indices]

    # Compute initial total cost
    distances = cdist(data, medoids)
    labels = np.argmin(distances, axis=1)
    total_cost = np.sum(np.min(distances, axis=1))

    for iteration in range(max_iter):
        best_swap = None
        best_cost_reduction = 0

        nearest_distances = np.min(distances, axis=1)
        second_nearest_distances = np.partition(distances, 1, axis=1)[:, 1]
        nearest_medoids = np.argmin(distances, axis=1)

        for xj_idx in range(n):
            if xj_idx in medoid_indices:
                continue  # Skip if already a medoid

            delta_TD = -nearest_distances.copy()

            for xo_idx in range(n):
                if xo_idx == xj_idx:
                    continue
                
                doj = np.linalg.norm(data[xo_idx] - data[xj_idx])
                current_medoid_idx = nearest_medoids[xo_idx]
                dn = nearest_distances[xo_idx]
                ds = second_nearest_distances[xo_idx]

                if current_medoid_idx < k:  # Ensure index is valid
                    delta_TD[current_medoid_idx] += min(doj, ds) - dn

                if doj < dn:
                    for mi in range(k):
                        if medoid_indices[mi] != current_medoid_idx:
                            delta_TD[mi] += doj - dn

            # Choose the best medoid index safely
            best_mi_idx = np.argmin(delta_TD[:k])  # Ensure within bounds
            best_mi = medoid_indices[best_mi_idx]

            if best_mi_idx >= k:  # Prevent out-of-bounds errors
                continue

            if delta_TD[best_mi_idx] < best_cost_reduction:
                best_cost_reduction = delta_TD[best_mi_idx]
                best_swap = (best_mi, xj_idx)

        # If no improvement, stop
        if best_swap is None or best_cost_reduction >= -tol:
            return medoids, total_cost, best_cost_reduction < -tol

        # Perform the best swap
        swap_idx = np.where(medoid_indices == best_swap[0])[0][0]
        medoid_indices[swap_idx] = best_swap[1]
        medoids = data[medoid_indices]

        # Update distances and total cost
        distances = cdist(data, medoids)
        total_cost += best_cost_reduction

    return medoids, total_cost, False  # Did not fully converge

def fastpam2_swap(data, k, max_iter=300, tol=1e-4, tau=0.5):
    """
    FastPAM2 algorithm for k-medoids clustering (multi-candidate SWAP step).

    Parameters:
        data (np.array): Normalized data points (n x d).
        k (int): Number of clusters (medoids).
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance for total cost reduction.
        tau (float): Tolerance parameter for deciding when to apply swaps.

    Returns:
        ClusteringResult: Named tuple with labels, cluster centers, and data.
    """
    n = data.shape[0]

    # Step 1: Initialize medoids randomly
    medoid_indices = np.random.choice(n, k, replace=True)
    medoids = data[medoid_indices]

    # Compute initial total cost
    distances = cdist(data, medoids)  # Compute distances to medoids
    labels = np.argmin(distances, axis=1)  # Assign clusters
    total_cost = np.sum(np.min(distances, axis=1))  # Compute TD

    for iteration in range(max_iter):
        # Step 2: Compute nearest and second-nearest medoid distances
        nearest_distances = np.min(distances, axis=1)
        second_nearest_distances = np.partition(distances, 1, axis=1)[:, 1]
        nearest_medoids = np.argmin(distances, axis=1)

        # Step 3: Initialize swap candidates
        best_cost_reduction = np.zeros(k)
        best_swaps = np.full(k, None, dtype=object)

        # Iterate over all non-medoids
        for xj_idx in range(n):
            if xj_idx in medoid_indices:
                continue  # Skip if it's already a medoid

            # Compute ∆TD vector for swapping xj into medoid set
            delta_TD = -nearest_distances.copy()

            for xo_idx in range(n):
                if xo_idx == xj_idx:
                    continue
                
                doj = np.linalg.norm(data[xo_idx] - data[xj_idx])  # Distance to new medoid
                current_medoid_idx = nearest_medoids[xo_idx]
                dn = nearest_distances[xo_idx]
                ds = second_nearest_distances[xo_idx]

                delta_TD[current_medoid_idx] += min(doj, ds) - dn  # Loss update

                if doj < dn:
                    for mi in range(k):
                        if mi == current_medoid_idx:
                            continue
                        delta_TD[mi] += doj - dn

            # Store best swap candidates
            for i in range(k):
                if delta_TD[i] < best_cost_reduction[i]:
                    best_cost_reduction[i] = delta_TD[i]
                    best_swaps[i] = xj_idx

        # Step 4: Execute best swaps
        while True:
            i = np.argmin(best_cost_reduction)
            if best_cost_reduction[i] >= 0:
                break  # Stop if no improvement is found

            # Perform the best swap
            medoid_indices[i] = best_swaps[i]
            medoids = data[medoid_indices]

            # Update distances
            distances = cdist(data, medoids)
            labels = np.argmin(distances, axis=1)
            total_cost += best_cost_reduction[i]

            # Reset the executed swap
            best_cost_reduction[i] = 0

            # Step 5: Recompute costs for remaining swaps
            for j in range(k):
                if best_cost_reduction[j] < 0:
                    new_delta_TD = 0
                    for xo_idx in range(n):
                        if xo_idx in medoid_indices:
                            continue
                        new_delta_TD += np.linalg.norm(data[xo_idx] - data[best_swaps[j]]) - np.linalg.norm(data[xo_idx] - medoids[j])

                    if new_delta_TD <= tau * best_cost_reduction[j]:
                        best_cost_reduction[j] = new_delta_TD
                    else:
                        best_cost_reduction[j] = 0  # Skip this swap

    return medoids, total_cost, best_cost_reduction < -tol

    #return ClusteringResult(labels=labels, cluster_centers=medoids, data=data)


def pam_clustering(data: np.array, k: int, build_function: Callable = None, swap_function: Callable = None):
    """Perform k-medoids clustering using PAM (BUILD and SWAP phases)."""
    print(f"Data shape at the start of pam_clustering: {data.shape}")
    
    if build_function is None: 
        medoids, cost = pam_build(data, k)
    else:
        medoids, cost = build_function(data, k)  
    
    print(f"Initial medoids: {medoids}")
    medoids = np.array(medoids).flatten()
    
    if medoids.ndim != 1:
        raise ValueError(f"Medoids should be 1D, but got shape {medoids.shape}")
    
    improved = True
    
    while improved:
        if swap_function is None or swap_function is pam_swap:
            medoids, cost, improved = pam_swap(data, medoids)
        else: 
            print(f"Before swap function, medoids: {medoids}, type: {type(medoids)}")
            medoids, cost, improved = swap_function(data, k)
    
    medoids = np.array(medoids, dtype=int).flatten()
    print(f"Final medoids (indices): {medoids}")
    
    medoids_ = data[medoids]
    
    if medoids_.ndim == 1:
        medoids_ = medoids_.reshape(1, -1)
        print(f"Reshaped medoids to 2D: {medoids_.shape}")
    
    print(f"Medoids after selecting data points: {medoids_.shape}")
    
    distances = cdist(data, medoids_)
    print(f"Distances shape: {distances.shape}")
    
    labels = np.argmin(distances, axis=1)
    print(f"Labels shape: {labels.shape}")
    
    return ClusteringResult(labels=labels, cluster_centers=medoids_, data=data)


#boring implementations from sklearn and python. likely faster than custom implementation. does not work, problems with imports
def kmedoids_sklearn(data: np.array, k: int):
    """K-Medoids clustering using scikit-learn-extra. https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html 

    Args:
        data (np.array): The dataset (n_samples, n_features)
        k (int): Number of clusters

    Returns:
        ClusteringResult: Named tuple with labels, cluster centers, and data
    """
    model = KMedoids(n_clusters=k, method='pam', random_state=42)
    model.fit(data)
    
    labels = model.labels_
    cluster_centers = model.cluster_centers_

    return ClusteringResult(labels=labels, cluster_centers=cluster_centers, data=data)

def kmedoids_lib(data: np.array, k: int):
    """K-Medoids clustering using the kmedoids library. https://pypi.org/project/kmedoids/ 

    Args:
        data (np.array): The dataset (n_samples, n_features)
        k (int): Number of clusters

    Returns:
        ClusteringResult: Named tuple with labels, cluster centers, and data
    """
    distance_matrix = np.linalg.norm(data[:, np.newaxis] - data, axis=2)  # Compute pairwise distances
    model = KMedoidsLib(distance_matrix, k)
    model.process()
    
    medoid_indices = np.array(model.get_medoids())
    labels = np.argmin(distance_matrix[:, medoid_indices], axis=1)
    cluster_centers = data[medoid_indices]

    return ClusteringResult(labels=labels, cluster_centers=cluster_centers, data=data)