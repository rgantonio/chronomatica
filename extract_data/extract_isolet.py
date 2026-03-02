# --------------------------------------------
# Copyright 2025 KU Leuven
# Ryan Antonio <ryan.antonio@esat.kuleuven.be
#
# Description:
# This program extracts data from the ISOLET data set.
# You can find this from:
# https://archive.ics.uci.edu/dataset/54/isolet
#
# Data set can be imported via the python packages.
# --------------------------------------------

import os
from tqdm import tqdm
from collections import defaultdict
from ucimlrepo import fetch_ucirepo
from extract_data_util import (download_and_extract, scale_to_255)

# Directory paths and links
dir_current = os.path.dirname(os.path.abspath(__file__))

dir_output = dir_current + "/../data/isolet/"
os.makedirs(dir_output, exist_ok=True)

# Fetch dataset
isolet = fetch_ucirepo(id=54)

# Data (as pandas dataframes)
X = isolet.data.features
y = isolet.data.targets


isolet_counts = defaultdict(int)
isolet_files = {
    d: open(os.path.join(dir_output, f"{d}.txt"), "w") for d in range(26)
}

# Iterate through the samples
for i in tqdm(range(len(y))):
    # Get vector list
    X_vector = X.iloc[i].to_numpy()
    X_vector_list = []
    for j in range(len(X_vector)):
        X_vector_list.append(scale_to_255(X_vector[j]))
    isolet_str = " ".join(map(str, X_vector_list))
    # Get target class
    y_class = int(y["class"].iloc[i]) - 1
    # Save vector to appropriate entry
    isolet_files[y_class].write(f"{isolet_str}\n")

for f in isolet_files.values():
    f.close()