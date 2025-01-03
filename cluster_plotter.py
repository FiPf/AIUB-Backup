
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from cluster_finder import unnormalize

class ClusterPlotter:
    """
    A class to handle 2D and 3D plotting of clustered data with customizable coloring.
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

    def clusters_2d_plot(self, title: str, color_scheme='viridis', c=None, point_size=5, show_centers=True, grid=True):
        """
        Plot the clusters in 2D with customizable coloring.
        """
        plt.figure(figsize=(10, 7))
        coloring = c if c is not None else self.normalized_data[:, 2]

        scatter = plt.scatter(self.normalized_data[:, 1], self.normalized_data[:, 0],
                              c=coloring, cmap=color_scheme, s=point_size)
        
        if show_centers:
            plt.scatter(self.cluster_centers[:, 1], self.cluster_centers[:, 0],
                        c='red', marker='X', s=100, label='Cluster Centers')

        plt.xlabel('RAAN (omega)')
        plt.ylabel('Inclination (i)')
        plt.title(title)
        plt.colorbar(scatter, label='Custom Color' if c is not None else 'Eccentricity')
        plt.legend()
        if grid:
            plt.grid(True)
        plt.show()

    def clusters_3d_plot(self, title: str, color_scheme='viridis', c=None, point_size=5, show_centers=True):
        """
        Plot the clusters in 3D with customizable coloring.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        coloring = c if c is not None else self.normalized_data[:, 2]

        scatter = ax.scatter(self.normalized_data[:, 1], self.normalized_data[:, 2], self.normalized_data[:, 0],
                             c=coloring, cmap=color_scheme, s=point_size)

        if show_centers:
            ax.scatter(self.cluster_centers[:, 1], self.cluster_centers[:, 2], self.cluster_centers[:, 0],
                       c='red', marker='X', s=100, label='Cluster Centers')

        ax.set_xlabel('RAAN (omega)')
        ax.set_ylabel('Eccentricity (e)')
        ax.set_zlabel('Inclination (i)')
        ax.set_title(title)

        plt.colorbar(scatter, label='Custom Color' if c is not None else 'Eccentricity')
        ax.legend()
        plt.show()

    def combined_clusters_2d_plot(self, other_data, other_labels, other_centers, title: str, point_size=5, grid=True):
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
            c=self.labels, cmap='autumn', s=point_size, label='5 mm data clusters', zorder=1
        )

        plt.scatter(
            self.cluster_centers[:, 1], self.cluster_centers[:, 0],
            c='red', marker='X', s=100, label='5mm cm data centers', zorder=3
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

        plt.xlabel('RAAN (omega)')
        plt.ylabel('Inclination (i)')
        plt.title(title)
        plt.colorbar(scatter_self, label='5mm , color according to cluster')
        plt.colorbar(scatter_other, label='10 cm , color according to cluster')
        plt.legend()
        if grid:
            plt.grid(True)
        plt.show()
    
    def combined_clusters_3d_plot(self, other_data, other_labels, other_centers, title: str, point_size=5, show_centers=True):
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
            c=self.labels, cmap='autumn', s=point_size, label='5mm data clusters', zorder=3
        )

        if show_centers:
            ax.scatter(
                self.cluster_centers[:, 1], self.cluster_centers[:, 2], self.cluster_centers[:, 0],
                c='red', marker='X', s=100, label='5mm data centers', zorder=4
            )

        ax.set_xlabel('RAAN (omega)')
        ax.set_ylabel('Eccentricity (e)')
        ax.set_zlabel('Inclination (i)')
        ax.set_title(title)

        fig.colorbar(scatter_self, ax=ax, shrink=0.5, label='5mm data')
        fig.colorbar(scatter_other, ax=ax, shrink=0.5, label='10 cm data')
        ax.legend()

        plt.show()

