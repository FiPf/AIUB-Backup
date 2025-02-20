#Principal Component Analyisis PCA: Our data has at least 6 dimensions (the 6 orbital elements), we can add more (see 26 attributes in proof)
#PCA finds the unit vector u in this p-dimensional subspace such that when the data is projected to u, the variance of the projected data
#is maximized. Maximizing the variance ensures that we pick the direction in which the data fluctuates the
#most. 
#See machine learning lecture. 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
import sys
import os
sys.path.append(os.path.abspath("..")) 
import getdata

file = os.path.join("..", "input", "stat_Master_05_geo_s1.det")
data = getdata.array_extender(file)

# Assuming your data is in a NumPy array (n_samples, 26)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # Standardize

pca = PCA(n_components=26)  # Compute all components
pca.fit(data_scaled)

# Explained variance ratio to see how much each PC contributes
explained_variance = pca.explained_variance_ratio_

# Loadings show how original features contribute to PCs
loadings = np.abs(pca.components_)


num_pc = np.argmax(np.cumsum(explained_variance) > 0.95) + 1  # Keep 95% variance
reduced_data = pca.transform(data_scaled)[:, :num_pc]