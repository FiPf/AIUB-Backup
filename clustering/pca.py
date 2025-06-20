import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
path_to_parent = os.path.abspath("..")
sys.path.append(path_to_parent)
import getdata

file = os.path.join("..", "input", "stat_Master_05_geo_s1.det")
data = np.array(getdata.array_extender(file))
print("Original data shape:", data.shape)

# Indices of the orbit-related features in your full feature list
# Original feature list indexing:
# 0: ID, 1: diameter, 2: factor, 3: source, 4: TCA, 5: TCA_RNG, 6: TCA_ALT, 7: TCA_RRT,
# 8: sem_major, 9: inc, 10: ecc, 11: arg_per, 12: raan, 13: true_lat, ..., 20: mag_obj, ...

orbit_feature_indices = [1, 8, 9, 10, 11, 12, 13, 20]

# Select only these features
data_orbit = data[orbit_feature_indices, :].T  # shape: (num_samples, num_orbit_features)
print("Orbit-related data shape:", data_orbit.shape)

# Updated feature names (in order)
feature_names = [
    "diameter", "sem_major", "inc", "ecc", "arg_per", "raan", "true_lat", "mag_obj"
]

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_orbit)

# PCA
pca = PCA(n_components=data_scaled.shape[1])
pca.fit(data_scaled)

explained_variance = pca.explained_variance_ratio_
loadings = np.abs(pca.components_)

# Reduce dimensions keeping 95% of variance
num_pc = np.argmax(np.cumsum(explained_variance) > 0.95) + 1
reduced_data = pca.transform(data_scaled)[:, :num_pc]

print("Explained variance:", explained_variance)
print("Num principal components:", num_pc)
print("Reduced data shape:", reduced_data.shape)

# Plot cumulative explained variance
plt.plot(np.cumsum(explained_variance), marker='o')
plt.axhline(0.95, color='r', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance (Orbit Features Only)")
plt.grid(True)
plt.show()

# Scatter plot of first two PCs
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Data projected to first two principal components (Orbit Features Only)")
plt.grid(True)
plt.show()

# Identify most influential features per PC
top_feature_indices = np.argmax(loadings, axis=1)
top_n_pcs = num_pc
top_features_by_pc = top_feature_indices[:top_n_pcs]

max_loadings = loadings.max(axis=0)  # max loading per feature across all PCs
sorted_indices = np.argsort(-max_loadings)  # descending order

print("Features ranked by max loading across all PCs:")
for idx in sorted_indices:
    print(f"{feature_names[idx]}: {max_loadings[idx]:.4f}")

print("Most influential feature names by PC:")
for i, feature_idx in enumerate(np.argmax(loadings, axis=1)):
    print(f"PC{i+1}: {feature_names[feature_idx]}")

df = pd.DataFrame(data_orbit, columns=feature_names)
sns.pairplot(df)
plt.show()
