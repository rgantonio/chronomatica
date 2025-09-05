"""
Python library for Vector Symbolic Architectures (VSA)
- This library adds VSA higher-level functions like item memory generation,
  associative search, confusion matrix generation, etc.
"""

import numpy as np
import vsa as vsa


# Generate item memory (IM) with orthogonal hypervectors
# The IM are in 2D matrices so we can use matrix operations later
def gen_orthogonal_im(num_items, hv_size, type="bipolar", int_min=0, int_max=255):
    """
    Generate an item memory with orthogonal hypervectors.
    Parameters:
    num_items (int): The number of items in the item memory.
    hv_size (int): The size of each hypervector.
    type (str): The type of hypervector. Can be 'bipolar', 'binary', 'real', or 'complex'.
    Returns:
    np.ndarray: The generated item memory.
    """
    im = np.zeros((num_items, hv_size), dtype=complex if type == "complex" else float)
    for i in range(num_items):
        im[i] = vsa.gen_hv(hv_size, type, int_min, int_max)
    return im


# Creates a confusion matrix based on a given matrix
# This can be used to evaluate the orthogonality of an item memory
# Take note that the similarity computation can differ
# E.g., for binary values, the XOR is better than cosine
def gen_confusion_matrix(im, sim_func=vsa.hv_ham):
    """
    Generate a confusion matrix based on the given item memory.
    Parameters:
    im (np.ndarray): The item memory matrix where each row is a hypervector.
    Returns:
    np.ndarray: The confusion matrix.
    """
    num_items = im.shape[0]
    confusion_matrix = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            confusion_matrix[i, j] = sim_func(im[i], im[j])
    return confusion_matrix
