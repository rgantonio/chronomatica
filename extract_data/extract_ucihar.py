# --------------------------------------------
# Copyright 2025 KU Leuven
# Ryan Antonio <ryan.antonio@esat.kuleuven.be
#
# Description:
# This program extracts data from the UCIHAR data set.
# You can find this from:
# https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
#
# Note that the data extraction has a series
# of zip files cascaded to one another.
# --------------------------------------------

import os
import zipfile
from tqdm import tqdm
from collections import defaultdict
from extract_data_util import (download_and_extract, scale_to_255)

# Directory paths and links
dir_current = os.path.dirname(os.path.abspath(__file__))

dir_temp_data = dir_current + "/ucihar_raw_data"
dir_temp_raw_data = dir_temp_data + "/UCI HAR Dataset"
dir_raw_train_data = dir_temp_raw_data + "/train"
dir_raw_test_data = dir_temp_raw_data + "/test"

dir_output_train = dir_current + "/../data/ucihar/train"
dir_output_test = dir_current + "/../data/ucihar/test"

url="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

# Download and extract raw data
download_and_extract(
    url=url,
    out_dir=dir_temp_data,
)

# Further extraction of file in UCIHAR
print("Further extraction of sub-zip...")
ucihar_zip = dir_temp_data + "/UCI HAR Dataset.zip"
with zipfile.ZipFile(ucihar_zip, "r") as zf:
    zf.extractall(dir_temp_data)

# Create where to dump the data
os.makedirs(dir_output_train, exist_ok=True)
os.makedirs(dir_output_test, exist_ok=True)

ucihar_trainX_dir = dir_raw_train_data + "/X_train.txt"
ucihar_trainY_dir = dir_raw_train_data + "/y_train.txt"
ucihar_testX_dir = dir_raw_test_data + "/X_test.txt"
ucihar_testY_dir = dir_raw_test_data + "/y_test.txt"

# Parameters
num_classes = 6
num_features = 561

# For tracking purposes only
class_train_counts = defaultdict(int)
class_test_counts = defaultdict(int)

# First extract all data
X_train = []
Y_train = []
X_test = []
Y_test = []

with open(ucihar_trainX_dir, "r") as trXf:
    for line in trXf:
        str_line = line.strip().split()
        float_line = [scale_to_255(float(s)) for s in str_line]
        X_train.append(float_line)

with open(ucihar_trainY_dir, "r") as trYf:
    for line in trYf:
        Y_train.append(int(line.strip()))

with open(ucihar_testX_dir, "r") as tsXf:
    for line in tsXf:
        str_line = line.strip().split()
        float_line = [scale_to_255(float(s)) for s in str_line]
        X_test.append(float_line)

with open(ucihar_testY_dir, "r") as tsYf:
    for line in tsYf:
        Y_test.append(int(line.strip()))

# Make sure to close the files
trXf.close()
trYf.close()
tsXf.close()
tsYf.close()

# Prepare files to be written to
ucihar_train_files = {
    d: open(os.path.join(dir_output_train, f"{d}.txt"), "w")
    for d in range(num_classes)
}

ucihar_test_files = {
    d: open(os.path.join(dir_output_test, f"{d}.txt"), "w")
    for d in range(num_classes)
}

# Iterate through train
for i in tqdm(range(len(X_train)), desc="Writing UCI-HAR train"):
    # Get vector list
    ucihar_str = " ".join(map(str, X_train[i]))
    # Get target class
    y_class = Y_train[i] - 1  # Adjust for zero-based index
    # Save vector to appropriate entry
    ucihar_train_files[y_class].write(f"{ucihar_str}\n")
    class_train_counts[y_class] += 1

# Iterate through test
for i in tqdm(range(len(X_test)), desc="Writing UCI-HAR test"):
    # Get vector list
    ucihar_str = " ".join(map(str, X_test[i]))
    # Get target class
    y_class = Y_test[i] - 1  # Adjust for zero-based index
    # Save vector to appropriate entry
    ucihar_test_files[y_class].write(f"{ucihar_str}\n")
    class_test_counts[y_class] += 1

for f in ucihar_train_files.values():
    f.close()

for f in ucihar_test_files.values():
    f.close()

# Some statistics
for d in range(num_classes):
    print(
        f"Class {d}: Train samples = {class_train_counts[d]}, Test samples = {class_test_counts[d]}"
    )

print("Total train samples:", sum(class_train_counts.values()))
print("Total test samples:", sum(class_test_counts.values()))