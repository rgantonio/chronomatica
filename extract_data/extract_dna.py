# --------------------------------------------
# Copyright 2025 KU Leuven
# Ryan Antonio <ryan.antonio@esat.kuleuven.be
#
# Description:
# This program extracts data from the DNA data set.
# You can find this from:
# https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences
#
# --------------------------------------------

import os
from tqdm import tqdm
from collections import defaultdict
from ucimlrepo import fetch_ucirepo

# Directory paths and links
dir_current = os.path.dirname(os.path.abspath(__file__))

dir_output = dir_current + "/../data/dna/"
os.makedirs(dir_output, exist_ok=True)

# Dataset fixed information
feature_dict = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
    "D": 5,
    "R": 6,
    "S": 7,
}

class_dict = {"EI": 0, "IE": 1, "N": 2}

def convert_dna_to_num(input_list):
    new_list = []
    for i in range(len(input_list)):
        new_list.append(feature_dict[input_list[i]])
    return new_list


# fetch dataset
dna = fetch_ucirepo(id=69)

# data (as pandas dataframes)
X = dna.data.features
y = dna.data.targets

num_features = len(X.iloc[0])
num_items = len(y["class"])

X_vector = X.iloc[0]

dna_counts = defaultdict(int)
dna_files = {d: open(os.path.join(dir_output, f"{d}.txt"), "w") for d in range(len(class_dict))}

for i in tqdm(range(len(y))):
    # Get vector list
    X_vector = convert_dna_to_num(list(X.iloc[i]))
    dna_str = " ".join(map(str, X_vector))
    # Get target class
    y_class = class_dict[y["class"].iloc[i]]
    # Save vector to appropriate entry
    dna_files[y_class].write(f"{dna_str}\n")

for f in dna_files.values():
    f.close()