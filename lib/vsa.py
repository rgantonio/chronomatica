"""
    Python library for Vector Symbolic Architectures (VSA)
"""

import numpy as np
from tqdm import tqdm

"""
    Basic VSA operations
"""
# Binary/bipolar random indexed method for generating hypervectors
# This works for binary and bipolar hypervectors only
def gen_rand_idx_hv(size, type="bipolar"):
    """
    Generate a random indexed hypervector of a specified type.

    Args:
        size (int): The size of the hypervector.
        type (str): The type of hypervector. Can be 'bipolar' or 'binary'.

    Returns:
        np.ndarray: The generated hypervector.
    """
    hv = np.arange(size)
    np.random.shuffle(hv)
    if type == "binary":
        return_hv = (hv >= size // 2).astype(int)
    elif type == "bipolar":
        return_hv = np.where(hv >= size // 2, 1, -1)
    else:
        raise ValueError("Unsupported hypervector type")
    return return_hv


# Generate hypervector of bipolar values
def gen_hv(size, type="bipolar", int_min=0, int_max=255):
    """
    Generate a hypervector of a specified type.
    Args:
        size (int): The size of the hypervector.
        type (str): The type of hypervector. Can be 'bipolar', 'binary', 'real', or 'complex'.

    Returns:
        np.ndarray: The generated hypervector.
    """
    if type == "bipolar":
        return gen_rand_idx_hv(size, type="bipolar")
    elif type == "binary":
        return gen_rand_idx_hv(size, type="binary")
    elif type == "integer":
        return np.random.randint(int_min, int_max, size=size)
    elif type == "real":
        return np.random.uniform(-1, 1, size=size)
    elif type == "complex":
        return np.random.uniform(-1, 1, size=size) + 1j * np.random.uniform(
            -1, 1, size=size
        )
    else:
        raise ValueError("Unsupported hypervector type")

# Randomly flip bits in a hypervector
def rand_flip_hv(hv, start_flips, end_flips, hv_type="binary"):
    """
    Randomly flip bits in a hypervector.
    Args:
        hv (np.ndarray): The hypervector to flip bits in.
        start_flips (int): The starting index for flipping.
        end_flips (int): The ending index for flipping.
        hv_type (str): The type of hypervector. Can be 'bipolar'
    Returns:
        np.ndarray: The hypervector with flipped bits.
    """
    if hv_type == "bipolar":
        hv[start_flips:end_flips] *= -1
    else:
        hv[start_flips:end_flips] ^= 1
    return hv

# Dot product of two hypervectors
def hv_dot(hv1, hv2):
    """
    Compute the dot product of two hypervectors.

    Args:
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

    Args:
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
def hv_ham(hv1, hv2, normalize=True):
    """
    Compute the Hamming distance between two hypervectors.

    Args:
        hv1 (np.ndarray): The first hypervector.
        hv2 (np.ndarray): The second hypervector.

    Returns:
        int: The Hamming distance between the two hypervectors.
    """
    if normalize:
        return np.sum(hv1 != hv2) / len(hv1)
    else:
        return np.sum(hv1 != hv2)


# Circular permutation of a hypervector
def hv_perm(hv, shift):
    """
    Perform a circular permutation on a hypervector.

    Args:
        hv (np.ndarray): The hypervector to permute.
        shift (int): The number of positions to shift.

    Returns:
        np.ndarray: The permuted hypervector.
    """
    return np.roll(hv, shift)  # Circular shift


# Adding hypervectors in a list with
# an sign magnitude to convert to bipolar
# and an optional binarization with configurable threshold for binarization
def hv_add(hvs, dont_flatten=False, sign_magnitude=False, threshold=None):
    """
    Add a list of hypervectors together.

    Args:
        hvs (list): A list of hypervectors to add.
        dont_flatten (bool): If True, return the summed vector without flattening.
        sign_magnitude (bool): If True, convert to bipolar using sign magnitude.
        threshold (float): Threshold for binarization.

    Returns:
        np.ndarray: The resulting hypervector after addition.
    """
    result = np.sum(hvs, axis=0)

    if dont_flatten:
        return result

    if sign_magnitude:
        result = np.sign(result)
        return result

    if threshold is not None:
        result = np.where(result >= threshold, 1, 0)
        return result


# Subtraction of two hypervectors
def hv_sub(hv1, hv2):
    """
    Subtract one hypervector from another.

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
        hvs (list): A list of hypervectors to XOR.

    Returns:
        np.ndarray: The resulting hypervector after XOR.
    """
    result = hvs[0]

    for hv in hvs[1:]:
        result = hv_xor(result, hv)

    return result

def binarize_hv(hv_a, threshold, hv_type="binary"):
    # Binarize depending on hv_type
    # If it's binary use a threshold for this
    if hv_type == "bipolar":
        hv_a = np.where(hv_a >= 0, 1, -1)
    else:
        hv_a = np.where(hv_a >= threshold, 1, 0)

    return hv_a

def norm_dist_hv(hv_a, hv_b, hv_type="binary"):
    # If binary we do hamming distance,
    # else we do cosine similarity
    if hv_type == "bipolar":
        hv_dot = np.dot(hv_a, hv_b)
        norm_a = np.linalg.norm(hv_a)
        norm_b = np.linalg.norm(hv_b)
        dist = hv_dot / (norm_a * norm_b)
    else:
        ham_dist = np.sum(np.bitwise_xor(hv_a, hv_b))
        dist = 1 - (ham_dist / hv_a.size)
    return dist
"""
    Functions for VSA models
"""

# Creates empty memory hypervector
def gen_empty_mem_hv(num_hv=1024, hv_dim=1024):
    """
    Generate an empty memory hypervector.
    Args:
        num_hv (int): The number of hypervectors.
        hv_dim (int): The dimension of each hypervector.
    Returns:
        np.ndarray: The empty memory hypervector.
    """
    return np.zeros((num_hv, hv_dim), dtype=int)


# Generate item memory (IM) with orthogonal hypervectors
def gen_orthogonal_im(num_items=1024, hv_size=1024, type="bipolar", int_min=0, int_max=255):
    """
    Generate an item memory with orthogonal hypervectors.
    Args:
        num_items (int): The number of items in the item memory.
        hv_size (int): The size of each hypervector.
        type (str): The type of hypervector. Can be 'bipolar', 'binary', 'real', or 'complex'.
    Returns:
        np.ndarray: The generated item memory.
    """
    im = np.zeros((num_items, hv_size), dtype=complex if type == "complex" else int)
    for i in range(num_items):
        im[i] = gen_hv(hv_size, type, int_min, int_max)
    return im

# Generate continuous item memory (CIM)
def gen_continuous_im(
    num_items=21,
    hv_size=1024,
    cim_max_is_ortho=True,
    hv_type="bipolar",
):

    # First initialize some seed HV
    # Calculate % number of flips
    if cim_max_is_ortho:
        num_flips = (hv_size // 2) // (num_items - 1)
    else:
        num_flips = hv_size // (num_items - 1)

    # Initialize empty matrix
    cim = gen_empty_mem_hv(num_items, hv_size)

    # Generate first seed HV
    cim[0] = gen_hv(hv_size, hv_type)

    # Iteratively generate other HVs
    for i in range(num_items - 1):
        cim[i + 1] = rand_flip_hv(
            cim[i], i * num_flips, (i + 1) * num_flips, hv_type=hv_type
        )
    return cim

# Creates a confusion matrix based on a given matrix
def gen_confusion_matrix(im, sim_func=hv_ham):
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

def prediction_idx(class_am, encoded_hv, hv_type="binary"):
    score_list = []
    for i in range(len(class_am)):
        score_list.append(norm_dist_hv(class_am[i], encoded_hv, hv_type=hv_type))

    predict_idx = np.argmax(score_list)

    return predict_idx

# Class for VSA models
class ModelVSA:
    def __init__(self, hv_size=1024, hv_type="bipolar", num_ortho_im=1024, num_cim=21, cim_max_is_ortho=True, class_list=None):
        # Base parameters
        self.hv_size = hv_size
        self.hv_type = hv_type
        self.num_ortho_im = num_ortho_im
        self.num_cim = num_cim
        self.cim_max_is_ortho = cim_max_is_ortho

        # Parameters that will be determined later
        self.class_list = class_list
        self.num_classes = len(class_list)

        # Some extra switches
        self.binraize_encode = False
        self.binarize_am = False

        # Generate list of item memories (iMs)
        self.ortho_im = gen_orthogonal_im(
            num_items=num_ortho_im,
            hv_size=hv_size,
            type=hv_type
        )

        # Generate list of CiM
        self.cim = gen_continuous_im(
            num_items=num_cim,
            hv_size=hv_size,
            cim_max_is_ortho=cim_max_is_ortho,
            hv_type=hv_type,
        )

        # Initialization of associative memories
        self.class_am = dict()
        self.class_am_frozen = dict()
        self.class_am_bin = dict()
        self.class_am_count = dict()

        # Some statistics for testing
        self.test_class_score = dict()
        self.test_class_accuracy = dict()
        self.model_accuracy = None

    # Main encoding function
    def encode(self, item_data):
        print("Empty encoding, please make sure to override this function in the subclass.")

    # Training function
    def train_model(self, X_train):
        for class_label in range(self.num_classes):
            data_len = len(X_train[class_label])

            # Non-binarized training
            for item_num in tqdm(range(data_len), desc=f"Training class {class_label}"):
                # Getting encodede HV
                encoded_vec = self.encode(X_train[class_label][item_num])
                # Fill in the appropriate class
                if self.class_am.get(class_label) is None:
                    self.class_am[class_label] = encoded_vec
                else:
                    self.class_am[class_label] += encoded_vec

            # Automatically compute binarized output
            threshold = data_len / 2
            self.class_am_bin[class_label] = binarize_hv(self.class_am[class_label], threshold, self.hv_type)

            # Setting the frozen class
            self.class_am_frozen[class_label] = np.copy(self.class_am[class_label])

            # Updating class number
            self.class_am_count[class_label] = data_len
        print("Training complete!")

    def retrain_model(self, X_train):
        for class_label in range(self.num_classes):
            data_len = len(X_train[class_label])

            # Retraining with binarized AM
            for item_num in tqdm(range(data_len), desc=f"Retraining class {class_label}"):
                # Getting encoded HV
                encoded_vec = self.encode(X_train[class_label][item_num])
                # Predicting first
                predict_label = prediction_idx(self.class_am_frozen, encoded_vec, hv_type=self.hv_type)

                # If incorrect we update the AMs
                if predict_label != class_label:
                    # Subtract from wrong class AM
                    self.class_am[predict_label] -= encoded_vec
                    self.class_am_count[predict_label] -= 1
                    # Add to correct class AM
                    self.class_am[class_label] += encoded_vec
                    self.class_am_count[class_label] += 1

            # Automatically compute binarized output
            threshold = self.class_am_count[class_label] / 2
            self.class_am_bin[class_label] = binarize_hv(self.class_am[class_label], threshold, self.hv_type)

        # For updating the frozen AM
        for class_label in range(self.num_classes):
            # Update frozen AM
            self.class_am_frozen[class_label] = np.copy(self.class_am[class_label])

        print("Retraining complete!")

    def test_model(self, X_test):
        correct_count = 0
        class_correct_count = 0
        total_count = 0

        for class_label in range(self.num_classes):
            data_len = len(X_test[class_label])
            class_correct_count = 0
            for item_num in tqdm(range(data_len), desc=f"Testing class {class_label}"):
                # Getting encoded HV
                encoded_vec = self.encode(X_test[class_label][item_num])
                # Compare with each class AM
                predict_label = prediction_idx(self.class_am_frozen, encoded_vec, hv_type=self.hv_type)

                if predict_label == class_label:
                    correct_count += 1
                    class_correct_count += 1
                total_count += 1

            self.test_class_score[class_label] = class_correct_count
            self.test_class_accuracy[class_label] = class_correct_count / data_len

        # Total score
        accuracy = correct_count / total_count
        self.model_accuracy = accuracy
        return accuracy

    def print_model_stats(self):
        print("")
        print("-----------------")
        print("Model Statistics:")
        print("-----------------")

        # Print internal parameters
        print(f"HV Size: {self.hv_size}")
        print(f"HV Type: {self.hv_type}")
        print(f"Number of Orthogonal IMs: {self.num_ortho_im}")
        print(f"Number of Continuous IMs: {self.num_cim}")
        print(f"Number of Classes: {self.num_classes}")

        # Print modes
        print(f"Binarize Encode: {self.binraize_encode}")
        print(f"Binarize AM: {self.binarize_am}")

        # Printing accuracies
        for class_label in range(self.num_classes):
            class_acc = self.test_class_accuracy.get(class_label, 0)
            print(f"Class {class_label} Accuracy: {class_acc*100:.2f}%")

        print(f"Overall Accuracy: {self.model_accuracy*100:.2f}%")
