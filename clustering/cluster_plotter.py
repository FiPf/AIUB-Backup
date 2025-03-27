import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.cm as cm

def clear_directory(directory: str): 
        """delete every file from the current directory (used to ensure that no plots/files are overwritten when rerunning the code)

        Args:
            directory (str): directory to clear
        """
        files = os.listdir(directory)
        for file_name in files:
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                    os.remove(file_path)

class ClusterPlotter:
    """
    A class to handle 2D and 3D plotting of clustered data.
    """
    def __init__(self, normalized_data, labels, cluster_centers):
        """
        Initialize the ClusterPlotter class.

        Parameters:
            normalized_data (np.ndarray): Normalized data points (NxM).
            labels (np.ndarray): Cluster labels for each data point.
            cluster_centers (np.ndarray): Cluster centers in normalized scale.
        """
        self.normalized_data = normalized_data
        self.labels = labels
        self.cluster_centers = cluster_centers

    def save_unique_plot(self,file_path: str, directory: str) -> str:
        """
        Helper function to ensure a unique filename in the given directory.
        If the filename already exists, append _1, _2, etc., to make it unique.
        Returns the new filename.
        """
        os.makedirs(directory, exist_ok=True)  # Ensure directory exists
        base_name = os.path.basename(file_path)
        base_path, extension = os.path.splitext(base_name)  # Split into base name and extension (e.g., .png)
        new_file_path = os.path.join(directory, base_name)
        
        count = 1
        while os.path.exists(new_file_path):  # Check if the file already exists
            new_file_path = os.path.join(directory, f"{base_path}_{count}{extension}") 
            count += 1 
        
        return new_file_path

    def clusters_2d_plot(self, title: str, save_name=None, color_scheme='Dark2', point_size=5, show_centers=True):
        """Plot the clusters in 2D with fixed coloring, sorting clusters by size.

        Args:
            title (str): Title of the plot.
            save_name (str, optional): File path to save the plot. If None, the plot is displayed.
            color_scheme (str, optional): Color scheme for the clusters. Defaults to 'Dark2'.
            point_size (int, optional): Size of the data points. Defaults to 5.
            show_centers (bool, optional): Show the centers of the clusters or not. Defaults to True.
        """
        plt.figure(figsize=(10, 7))

        # Sort clusters by size, ignoring noise (-1)
        unique_labels, counts = np.unique(self.labels, return_counts=True)

        # Separate noise and valid clusters
        is_noise = unique_labels == -1
        valid_labels = unique_labels[~is_noise]
        valid_counts = counts[~is_noise]

        # Sort clusters by size (descending)
        sorted_indices = np.argsort(-valid_counts)
        sorted_labels = valid_labels[sorted_indices]

        # Generate colormap with a fixed number of colors, excluding red
        color_map = cm.get_cmap(color_scheme, len(sorted_labels))

        # Assign colors based on sorted cluster order
        label_to_color = {label: color_map(i) for i, label in enumerate(sorted_labels)}
        label_to_color[-1] = (1, 0, 0, 1)  # Noise is always red (RGBA format)

        # Apply colors (fixing ValueError)
        coloring = [label_to_color[label] for label in self.labels]
        coloring = np.array(coloring, dtype=object)  # Ensure NumPy handles lists correctly

        # Scatter plot of data points
        plt.scatter(self.normalized_data[:, 1], self.normalized_data[:, 0], c=coloring, s=point_size)

        # Plot cluster centers
        if show_centers and hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
            plt.scatter(self.cluster_centers[:, 1], self.cluster_centers[:, 0], c='black', marker='X', s=100, label='Cluster Centers')

        # Fixed plot settings
        plt.xlabel('RAAN [°]')
        plt.ylabel('Inclination [°]')
        plt.title(title)
        plt.grid(True)

        # Save or show the plot
        if save_name is not None:
            unique_save_path = self.save_unique_plot(save_name, os.path.dirname(save_name))
            plt.savefig(unique_save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {unique_save_path}")
            plt.close()
        else:
            plt.show()



    def clusters_3d_plot(self, title: str, color_scheme='Dark2', c=None, point_size=5, show_centers=True):
        """Plot the clusters in 3D with customizable coloring.

        Args:
            title (str): Title of the plot. 
            color_scheme (str, optional): _description_. Defaults to 'Dark2'.
            c (_type_, optional): _description_. Defaults to None.
            point_size (int, optional): _description_. Defaults to 5.
            show_centers (bool, optional): _description_. Defaults to True.
        """        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        coloring = c if c is not None else self.normalized_data[:, 2]

        scatter = ax.scatter(self.normalized_data[:, 1], self.normalized_data[:, 2], self.normalized_data[:, 0],
                             c=coloring, cmap=color_scheme, s=point_size)

        if show_centers:
            ax.scatter(self.cluster_centers[:, 1], self.cluster_centers[:, 2], self.cluster_centers[:, 0],
                       c='red', marker='X', s=100, label='Cluster Centers')

        ax.set_xlabel('RAAN [°]')
        ax.set_ylabel('Eccentricity (e)')
        ax.set_zlabel('Inclination [°]')
        ax.set_title(title)

        plt.colorbar(scatter, label='Custom Color' if c is not None else 'Eccentricity')
        ax.legend()
        plt.show()

    def combined_clusters_2d_plot(self, other_data, other_labels, other_centers, title: str, size_in_mm: int, point_size=5, grid=True):
        """
        Plot this dataset and another dataset (2D) with different color schemes and cluster centers.

        Parameters:
            other_data (array-like): 2D coordinates of another dataset.
            other_labels (array-like): Cluster labels for the other dataset.
            other_centers (array-like): Cluster centers for the other dataset.
            title (str): Title of the plot.
            point_size (int): Size of the points in the scatter plot.
            grid (bool): Whether to show a grid.
        """
        plt.figure(figsize=(10, 7))

        # Plot the current dataset (5mm) first
        scatter_self = plt.scatter(
            self.normalized_data[:, 1], self.normalized_data[:, 0],
            c=self.labels, cmap='autumn', s=point_size, label=f'{size_in_mm}mm data clusters', zorder=1
        )

        plt.scatter(
            self.cluster_centers[:, 1], self.cluster_centers[:, 0],
            c='red', marker='X', s=100, label=f'{size_in_mm}mm cm data centers', zorder=3
        )

        # Plot the other dataset (10 cm) on top
        scatter_other = plt.scatter(
            other_data[:, 1], other_data[:, 0],
            c=other_labels, cmap='summer', s=point_size, label='10 cm data clusters', zorder=2
        )

        plt.scatter(
            other_centers[:, 1], other_centers[:, 0],
            c='blue', marker='X', s=100, label='10 cm data centers', zorder=4
        )

        plt.xlabel('RAAN [°]')
        plt.ylabel('Inclination [°]')
        plt.title(title)
        plt.colorbar(scatter_self, label=f'{size_in_mm}mm , color according to cluster')
        plt.colorbar(scatter_other, label='10 cm , color according to cluster')
        plt.legend()
        if grid:
            plt.grid(True)
        plt.show()
    
    def combined_clusters_3d_plot(self, other_data, other_labels, other_centers, title: str, size_in_mm: int, point_size=5, show_centers=True):
        """
        Plot this dataset and another dataset (3D) with different color schemes and cluster centers.

        Parameters:
            other_data (array-like): 3D coordinates of another dataset.
            other_labels (array-like): Cluster labels for the other dataset.
            other_centers (array-like): Cluster centers for the other dataset.
            title (str): Title of the plot.
            point_size (int): Size of the points in the scatter plot.
            show_centers (bool): Whether to show cluster centers.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the other dataset (10cm data) first
        scatter_other = ax.scatter(
            other_data[:, 1], other_data[:, 2], other_data[:, 0],
            c=other_labels, cmap='summer', s=point_size, label='10cm data clusters', zorder=1
        )

        if show_centers:
            ax.scatter(
                other_centers[:, 1], other_centers[:, 2], other_centers[:, 0],
                c='blue', marker='X', s=100, label='10 cm data centers', zorder=2
            )

        # Plot the current dataset (5mm data) on top
        scatter_self = ax.scatter(
            self.normalized_data[:, 1], self.normalized_data[:, 2], self.normalized_data[:, 0],
            c=self.labels, cmap='autumn', s=point_size, label=f'{size_in_mm}mm data clusters', zorder=3
        )

        if show_centers:
            ax.scatter(
                self.cluster_centers[:, 1], self.cluster_centers[:, 2], self.cluster_centers[:, 0],
                c='red', marker='X', s=100, label=f'{size_in_mm}mm data centers', zorder=4
            )

        ax.set_xlabel('RAAN (omega)')
        ax.set_ylabel('Eccentricity (e)')
        ax.set_zlabel('Inclination (i)')
        ax.set_title(title)

        fig.colorbar(scatter_self, ax=ax, shrink=0.5, label=f'{size_in_mm}mm data')
        fig.colorbar(scatter_other, ax=ax, shrink=0.5, label='10 cm data')
        ax.legend()

        plt.show()