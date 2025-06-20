import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

def plot_anova_f_values(cluster_data, labels, output_folder, filename="anova_f_values.png", title = ""):
    """
    Generate and save a bar plot of ANOVA F-values between each feature and cluster labels.

    Args:
        cluster_data (ClusterData): Namedtuple with fields inc, raan, perigee, ecc, mag.
        labels (array-like, shape (N,)): Cluster labels for each data point.
        output_folder (str): Folder to save the ANOVA plot image.
        filename (str, optional): Name of the output file. Defaults to "anova_f_values.png".
    """
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)

    # Create feature DataFrame
    df = pd.DataFrame({
        "inclination": cluster_data.inc,
        "RAAN": cluster_data.raan,
        "Perigee": cluster_data.perigee,
        "eccentricity": cluster_data.ecc,
        "magnitude": cluster_data.mag
    })

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    # Compute ANOVA F-values
    f_vals, _ = f_classif(X_scaled, labels)
    features = df.columns.tolist()

    # Plot
    plt.figure(figsize=(6, 4))
    sns.barplot(x=f_vals, y=features, palette="crest")
    plt.xlabel("ANOVA F-value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_co_membership(labels, output_folder, filename="co_membership.png"):
    """
    Generate and save a co-membership heatmap for a given clustering assignment.

    Args:
        labels (array-like, shape (N,)): Cluster labels for each data point.
        output_folder (str): Folder to save the heatmap image.
        filename (str, optional): Name of the output file. Defaults to "co_membership.png".
    """
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)

    labels = np.array(labels)
    N = len(labels)
    # Create an NxN co-membership matrix: 1 if same cluster, else 0
    co_matrix = np.zeros((N, N), dtype=int)
    for i in range(N):
        co_matrix[i] = (labels == labels[i]).astype(int)

    plt.figure(figsize=(8, 6))
    sns.heatmap(co_matrix, cmap="viridis", cbar=False)
    plt.title("Co-membership Heatmap")
    plt.xlabel("Point Index")
    plt.ylabel("Point Index")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_correlation_heatmap(cluster_data, output_folder, filename="correlation_heatmap.png"):
    """
    Generate and save a 5×5 correlation heatmap for features in ClusterData.

    Args:
        cluster_data (ClusterData): Namedtuple with fields inc, raan, perigee, ecc, mag (arrays of shape (N,)).
        output_folder (str): Folder to save the heatmap image.
        filename (str, optional): Name of the output file. Defaults to "correlation_heatmap.png".
    """
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)

    # Build a DataFrame with all five features
    df = pd.DataFrame({
        "inclination": cluster_data.inc,
        "RAAN": cluster_data.raan,
        "Perigee": cluster_data.perigee,
        "eccentricity": cluster_data.ecc,
        "magnitude": cluster_data.mag
    })

    corr = df.corr(method="pearson")  # 5×5 correlation matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_mutual_information(cluster_data, labels, output_folder, filename="mutual_information.png"):
    """
    Generate and save a mutual information bar plot between each feature and the cluster labels.

    Args:
        cluster_data (ClusterData): Namedtuple with fields inc, raan, oerigee, ecc, mag.
        labels (array-like, shape (N,)): Cluster labels for each data point.
        output_folder (str): Folder to save the MI plot image.
        filename (str, optional): Name of the output file. Defaults to "mutual_information.png".
    """
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)

    # Build feature matrix
    df = pd.DataFrame({
        "inclination": cluster_data.inc,
        "RAAN": cluster_data.raan,
        "Perigee": cluster_data.perigee,
        "eccentricity": cluster_data.ecc,
        "magnitude": cluster_data.mag
    })

    # Standardize features (recommended for MI)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    mi_vals = mutual_info_classif(X_scaled, labels, discrete_features=False, random_state=0)
    features = df.columns.tolist()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=mi_vals, y=features, palette="magma")
    plt.xlabel("Mutual Information")
    plt.title("MI: Features vs. Cluster Labels")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
