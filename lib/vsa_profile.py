"""
Python library for Vector Symbolic Architectures (VSA)
- This library contains functions for profiling or measuring
  statistics of various cases in VSA.
"""

import numpy as np
import matplotlib.pyplot as plt


# Z-score normalization
def zscore_normalize(x):
    """
    Z-score normalization of a numpy array.
    Parameters:
    x (np.ndarray): The input array to normalize.
    Returns:
    np.ndarray: The normalized array.
    """
    return (x - np.mean(x)) / np.std(x)


# Min-max normalization
def minmax_normalize(x, min_val=0, max_val=1):
    """
    Min-max normalization of a numpy array.
    Parameters:
    x (np.ndarray): The input array to normalize.
    Returns:
    np.ndarray: The normalized array.
    """
    return (x - min_val) / (max_val - min_val)


# Extract upper triangle without diagnal
def extract_upper_tri_no_diag(x):
    """
    Extract the upper triangular part of a square matrix, excluding the diagonal.
    Parameters:
    x (np.ndarray): The input square matrix.
    Returns:
    np.ndarray: The extracted upper triangular values as a 1D array.
    """
    return x[np.triu_indices(x.shape[0], k=1)]


# Measure mean and variance
def measure_mean_var(x):
    """
    Measure the mean and variance of a numpy array.
    Parameters:
    x (np.ndarray): The input array.
    Returns:
    tuple: A tuple containing the mean and variance of the array.
    """
    return np.mean(x), np.var(x)


# Plotting the confusion matrix
def plot_confmap(
    conf_matrix, vmin=0, vmax=100, cmap="Blues", figsize=(6, 5), dpi=120, decimals=2
):
    """
    np.ndarray: The normalized array.
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# Plotting the confusion matrix
def plot_confmap(
    conf_matrix,
    vmin=0,
    vmax=100,
    cmap="Blues",
    figsize=(6, 5),
    dpi=120,
    decimals=2,
    annotations=True,
):
    """
    Plot a confusion matrix as a heatmap.
    Parameters:
    conf_matrix (np.ndarray): The confusion matrix to plot.
    vmin (float): Minimum value for colormap scaling.
    vmax (float): Maximum value for colormap scaling.
    cmap (str): Colormap to use for the heatmap.
    figsize (tuple): Figure size.
    dpi (int): Dots per inch for the figure.
    decimals (int): Number of decimal places to round the annotations.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Heatmap
    data = ax.imshow(
        conf_matrix, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax
    )

    # Colorbar
    plt.colorbar(data, ax=ax)

    # Axes: use indices as ticks
    ax.set_xticks(np.arange(conf_matrix.shape[1]))
    ax.set_yticks(np.arange(conf_matrix.shape[0]))
    ax.set_xlabel("HV Index")
    ax.set_ylabel("HV Index")
    ax.set_title("Confusion Matrix Heatmap")

    # Annotate each cell
    if annotations:
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    str(np.round(conf_matrix[i, j], decimals)),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if conf_matrix[i, j] > conf_matrix.max() / 2
                        else "black"
                    ),
                )

    plt.tight_layout()
    plt.show()
