import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

    from matplotlib.lines import Line2D

    def clusters_2d_plot(self, title: str, save_name=None, point_size=5, show_centers=True):
        plt.figure(figsize=(10, 7))

        # Sort clusters by size, ignoring noise (-1)
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        is_noise = unique_labels == -1
        valid_labels = unique_labels[~is_noise]
        valid_counts = counts[~is_noise]
        sorted_indices = np.argsort(-valid_counts)
        sorted_labels = valid_labels[sorted_indices]

        # Extreme colors palette
        extreme_colors = [
            "#026D02", "#00005F", '#FFFF00', "#A148A1", '#00FFFF',
            '#FF8000', '#8000FF', "#F157A4", "#04FF00", '#804000', '#000000',
            '#808080', "#922C2C", "#FF00BF", "#348FFE", "#6F496D", "#ECEC57", "#510051", "#005151",
            "#ED9366", '#8000FF', '#00FF80', '#804000', '#000000',
            "#2D3DB8"
        ]

        # Assign colors (cycle if more clusters than colors)
        label_to_color = {}
        for i, label in enumerate(sorted_labels):
            label_to_color[label] = extreme_colors[i % len(extreme_colors)]
        label_to_color[-1] = (1, 0, 0, 1)  # Noise in red

        coloring = [label_to_color[label] for label in self.labels]

        # Scatter plot
        plt.scatter(self.normalized_data[:, 1], self.normalized_data[:, 0], c=coloring, s=point_size)

        # Plot cluster centers
        if show_centers and hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
            plt.scatter(self.cluster_centers[:, 1], self.cluster_centers[:, 0],
                        c='black', marker='X', s=100, label='Cluster Centers')

        # Enlarged labels and title
        label_fontsize = 20
        title_fontsize = 18
        tick_fontsize = 16
        legend_fontsize = 16

        plt.xlabel(r'$\Omega$ [°]', fontsize=label_fontsize)
        plt.ylabel(r'$i$ [°]', fontsize=label_fontsize)
        plt.ylim(0, 22)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(True)

        # --- Legend with only numbers and title "Cluster" ---
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                markerfacecolor=label_to_color[label],
                markersize=20, label=f"{i+1}")   # only number
            for i, label in enumerate(sorted_labels)
        ]
        if -1 in self.labels:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=(1, 0, 0, 1),
                    markersize=20, label="Noise")
            )
        plt.legend(handles=legend_elements, title="Cluster",
                bbox_to_anchor=(1.05, 1), loc='upper left',
                borderaxespad=0., fontsize=legend_fontsize, title_fontsize=legend_fontsize)

        # Save or show
        if save_name is not None:
            unique_save_path = self.save_unique_plot(save_name, os.path.dirname(save_name))
            plt.savefig(unique_save_path, bbox_inches='tight', transparent=True)
            print(f"Plot saved as: {unique_save_path}")
            plt.close()
        else:
            plt.show()



    def clusters_3d_plot(self, title: str, save_name=None, color_scheme='Dark2', point_size=5, show_centers=True, feature_names=None, reverse_third_axis = False):
        """Plot the clusters in 3D with fixed coloring, sorting clusters by size.

        Args:
            title (str): Title of the plot.
            save_name (str, optional): File path to save the plot. If None, the plot is displayed.
            color_scheme (str, optional): Color scheme for the clusters. Defaults to 'Dark2'.
            point_size (int, optional): Size of the data points. Defaults to 5.
            show_centers (bool, optional): Show the centers of the clusters or not. Defaults to True.
            feature_names (list[str], optional): Names of the features in the data. Defaults to ['inc', 'raan', 'ecc'].
        """
        if feature_names is None:
            feature_names = ['inclination [°]', 'raan [°]', 'eccentricity [°]']  # Default ordering

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Sort clusters by size, ignoring noise (-1)
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        is_noise = unique_labels == -1
        valid_labels = unique_labels[~is_noise]
        valid_counts = counts[~is_noise]

        sorted_indices = np.argsort(-valid_counts)
        sorted_labels = valid_labels[sorted_indices]

        # Two color maps to avoid repetition
        color_map_1 = cm.get_cmap(color_scheme, len(sorted_labels))
        color_map_2 = cm.get_cmap('tab20c', len(sorted_labels))

        # Warn if too many clusters
        max_colors_1 = color_map_1.N
        max_colors_2 = color_map_2.N
        if len(sorted_labels) > max_colors_1 + max_colors_2:
            print(f"Warning: Number of clusters ({len(sorted_labels)}) exceeds combined color palette size ({max_colors_1 + max_colors_2}). Colors may repeat.")

        label_to_color = {}
        for i, label in enumerate(sorted_labels):
            if i % 2 == 0:
                label_to_color[label] = color_map_1(i // 2)
            else:
                label_to_color[label] = color_map_2(i // 2)

        label_to_color[-1] = (1, 0, 0, 1)  # Noise in red

        # Apply colors to each point
        coloring = [label_to_color[label] for label in self.labels]

        # Plot points (x=feature 1, y=feature 2, z=feature 0)
        ax.scatter(self.normalized_data[:, 1], self.normalized_data[:, 2], self.normalized_data[:, 0],
                c=coloring, s=point_size)

        if show_centers and hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
            ax.scatter(self.cluster_centers[:, 1], self.cluster_centers[:, 2], self.cluster_centers[:, 0],
                    c='black', marker='X', s=100, label='Cluster Centers')

        # Dynamic axis labels
        ax.set_xlabel(f"{feature_names[1]}")
        ax.set_ylabel(f"{feature_names[2]})")
        ax.set_zlabel(f"{feature_names[0]}")
        ax.set_title(title)

        if reverse_third_axis:
            ax.invert_yaxis()

        # Add legend
        legend_elements = [Patch(facecolor=label_to_color[label], label=f"Cluster {i+1}") for i, label in enumerate(sorted_labels)]
        if -1 in self.labels:
            legend_elements.append(Patch(facecolor=(1, 0, 0, 1), label="Noise"))
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        if save_name is not None:
            unique_save_path = self.save_unique_plot(save_name, os.path.dirname(save_name))
            plt.savefig(unique_save_path, dpi=300, bbox_inches='tight')
            print(f"3D plot saved as: {unique_save_path}")
            plt.close()
        else:
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

def save_table_as_image(df, filename: str, title: str=None, fontsize: int=10):
    """save a table from a dataframe as image

    Args:
        df (dataframe): dataframe that should be saved as image
        filename (str): filename, where to save the final image
        title (str, optional): title of the image. Defaults to None.
        fontsize (int, optional): Fontsize of the table. Defaults to 10.
    """
    fig, ax = plt.subplots(figsize=(len(df.columns)*2, len(df)*0.6 + 1))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.2, 1.2)

    if title:
        plt.title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
