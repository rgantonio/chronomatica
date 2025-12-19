#------------------------------------
# MNIST hand-written digit recognition with VSA
#------------------------------------

# Parameters
import os
import sys
import numpy as np

# Path directories
curr_dir = os.getcwd()
lib_path = curr_dir + "/../../lib"
extract_path = curr_dir + "/../../extract_data"
data_path = curr_dir + "/../../data"
dir_src_data = data_path + "/mnist_bin"

# Appending other paths
sys.path.append(lib_path)
sys.path.append(extract_path)

import vsa
from extract_data_util import download_and_extract, split_data, read_data

#------------------------------------
# Download and prepare train and test data
#------------------------------------
url = "https://github.com/rgantonio/chronomatica/releases/download/mnist_dataset_v1.0/chronomatica_mnist_bin.tar.gz"

download_and_extract(
    url=url,
    out_dir=data_path,
    delete_archive=True,
)

# Set class list
class_list = [0,1,2,3,4,5,6,7,8,9]

# Read data
X_data = read_data(class_list, dir_src_data)

# Train and test split
train_test_split = 0.6
train_valid_split = 0.75

X_train_set, X_test_set = split_data(X_data, class_list, split_percent=0.6)
X_train_set_src, X_valid_set = split_data(X_train_set, class_list, split_percent=0.75)

#------------------------------------
# Creating VSA model
#------------------------------------
class digitVSA(vsa.ModelVSA):
    def encode(self, item_data):
        # Feature length
        item_len = len(item_data)
        # Threshold for binarization
        threshold = item_len // 2
        # Encode hypervector
        encoded_vec = np.zeros(self.hv_size,dtype=int)
        for i in range(item_len):
            if(item_data[i] == 0):
                encoded_vec += self.ortho_im[i]
            else:
                encoded_vec += vsa.hv_perm(self.ortho_im[i],1)
        # Binarization
        encoded_vec = vsa.binarize_hv(encoded_vec, threshold, self.hv_type)
        return encoded_vec


# Create the VSA model with target parameters
digit_model = digitVSA(
    hv_size=1024,
    class_list = class_list
    )

# Train the model
digit_model.train_model(X_train_set_src)

# Retraining the model
digit_model.retrain_model(X_valid_set)

# Test the model
digit_model.test_model(X_test_set)

# Print some statistics
digit_model.print_model_stats()