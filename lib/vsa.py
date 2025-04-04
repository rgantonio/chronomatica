'''
  Python library for Vector Symbolic Architectures (VSA)
'''

import numpy as np

# Generate hypervector of bipolar values
def gen_hv(size, type='bipolar'):
    """
    Generate a hypervector of a specified type.
    Parameters:
    size (int): The size of the hypervector.
    type (str): The type of hypervector. Can be 'bipolar', 'binary', 'real', or 'complex'.
    Returns:
    np.ndarray: The generated hypervector.
    """
    if type == 'bipolar':
        return np.random.choice([-1, 1], size=size)
    elif type == 'binary':
        return np.random.choice([0, 1], size=size)
    elif type == 'real':
        return np.random.uniform(-1, 1, size=size)
    elif type == 'complex':
        return np.random.uniform(-1, 1, size=size) + 1j * np.random.uniform(-1, 1, size=size)
    else:
        raise ValueError("Unsupported hypervector type")
  
  # Dot product of two hypervectors
def hv_dot(hv1, hv2):
    """
    Compute the dot product of two hypervectors.
    Parameters:
    hv1 (np.ndarray): The first hypervector.
    hv2 (np.ndarray): The second hypervector.
    Returns:
    float: The dot product of the two hypervectors.
    """
    return np.dot(hv1, hv2)

# Cosine similarity between two hypervectors
def hv_cos(hv1, hv2):
    """
    Compute the cosine similarity between two hypervectors.
    Parameters:
    hv1 (np.ndarray): The first hypervector.
    hv2 (np.ndarray): The second hypervector.
    Returns:
    float: The cosine similarity between the two hypervectors.
    """
    dot_product = hv_dot(hv1, hv2)
    norm_hv1 = np.linalg.norm(hv1)
    norm_hv2 = np.linalg.norm(hv2)
    
    if norm_hv1 == 0 or norm_hv2 == 0:
        return 0.0
    else:
        return dot_product / (norm_hv1 * norm_hv2)
    
# Hamming distance between two hypervectors
def hv_ham(hv1, hv2):
    """
    Compute the Hamming distance between two hypervectors.
    Parameters:
    hv1 (np.ndarray): The first hypervector.
    hv2 (np.ndarray): The second hypervector.
    Returns:
    int: The Hamming distance between the two hypervectors.
    """
    return np.sum(hv1 != hv2)

# Circular permutation of a hypervector
def hv_perm(hv, shift):
    """
    Perform a circular permutation on a hypervector.
    Parameters:
    hv (np.ndarray): The hypervector to permute.
    shift (int): The number of positions to shift.
    Returns:
    np.ndarray: The permuted hypervector.
    """
    return np.roll(hv, shift)  # Circular shift

# Adding hypervectors in a list with
# an sign magnitude to convert to bipolar
# and an optional binarization with configurable threshold for binarization
def hv_add(hvs, sign_magnitude=False, threshold=None):
    """
    Add a list of hypervectors together.
    Parameters:
    hvs (list): A list of hypervectors to add.
    sign_magnitude (bool): If True, convert to bipolar using sign magnitude.
    threshold (float): Threshold for binarization.
    Returns:
    np.ndarray: The resulting hypervector after addition.
    """
    result = np.sum(hvs, axis=0)
    
    if sign_magnitude:
        result = np.sign(result)
    
    if threshold is not None:
        result = np.where(result >= threshold, 1, 0)
    
    return result


# Subtraction of two hypervectors
def hv_sub(hv1, hv2):
    """
    Subtract one hypervector from another.
    Parameters:
    hv1 (np.ndarray): The first hypervector.
    hv2 (np.ndarray): The second hypervector.
    Returns:
    np.ndarray: The resulting hypervector after subtraction.
    """
    return hv1 - hv2

# Multiplication of two hypervectors
def hv_mult(hv1, hv2):
    """
    Multiply two hypervectors element-wise.
    Parameters:
    hv1 (np.ndarray): The first hypervector.
    hv2 (np.ndarray): The second hypervector.
    Returns:
    np.ndarray: The resulting hypervector after multiplication.
    """
    return np.multiply(hv1, hv2)

# Element-wise XOR of two hypervectors
def hv_xor(hv1, hv2):
    """
    Compute the element-wise XOR of two hypervectors.
    Parameters:
    hv1 (np.ndarray): The first hypervector.
    hv2 (np.ndarray): The second hypervector.
    Returns:
    np.ndarray: The resulting hypervector after XOR.
    """
    return np.bitwise_xor(hv1, hv2)

# Multiplication of all hypervectors in a list
def hv_mult_list(hvs):
    """
    Multiply a list of hypervectors together.
    Parameters:
    hvs (list): A list of hypervectors to multiply.
    Returns:
    np.ndarray: The resulting hypervector after multiplication.
    """
    result = hvs[0]
    
    for hv in hvs[1:]:
        result = hv_mult(result, hv)
    
    return result

# XOR of all hypervectors in a list
def hv_xor_list(hvs):
    """
    Compute the XOR of a list of hypervectors.
    Parameters:
    hvs (list): A list of hypervectors to XOR.
    Returns:
    np.ndarray: The resulting hypervector after XOR.
    """
    result = hvs[0]
    
    for hv in hvs[1:]:
        result = hv_xor(result, hv)
    
    return result