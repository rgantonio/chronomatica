# --------------------------------------------
# Copyright 2024 KU Leuven
# Ryan Antonio <ryan.antonio@esat.kuleuven.be
#
# Description:
# Use this program just to extract a few samples of the MNIST dataset
# However, after extraction we will save it as released assets
# --------------------------------------------

import os
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from collections import defaultdict

# Directory paths
dir_current = os.path.dirname(os.path.abspath(__file__))
dir_bin_data = dir_current + "/../data/mnist_bin"
dir_uint_data = dir_current + "/../data/mnist_uint"
dir_int_data = dir_current + "/../data/mnist_int"

# Make directory paths
os.makedirs(dir_bin_data, exist_ok=True)
os.makedirs(dir_uint_data, exist_ok=True)
os.makedirs(dir_int_data, exist_ok=True)

# Parameters
samples = 1000
num_classes = 10

# Download and load MNIST training data
mnist = datasets.MNIST(
    root="./mnist_raw_data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

# For tracking purposes only
digit_counts = defaultdict(int)

# Opening file handlers
bin_digit_files = {
    d: open(os.path.join(dir_bin_data, f"{d}.txt"), "w") for d in range(num_classes)
}

uint_digit_files = {
    d: open(os.path.join(dir_uint_data, f"{d}.txt"), "w") for d in range(num_classes)
}

int_digit_files = {
    d: open(os.path.join(dir_int_data, f"{d}.txt"), "w") for d in range(num_classes)
}

# Collect and write samples
for img, label in tqdm(mnist, desc="Processing MNIST"):
    # Convert image to different formats
    bin_img = (img > 0).to(torch.uint8).view(-1).tolist()
    uint_img = (img * 255).to(torch.uint8).view(-1).tolist()
    int_img = (img * 127).round().clamp(-128, 127).to(torch.int8).view(-1).tolist()

    # Make them into strings
    bin_str = " ".join(map(str, bin_img))
    uint_str = " ".join(map(str, uint_img))
    int_str = " ".join(map(str, int_img))

    # Write to respective files
    bin_digit_files[label].write(f"{bin_str}\n")
    uint_digit_files[label].write(f"{uint_str}\n")
    int_digit_files[label].write(f"{int_str}\n")

    # Increment count
    digit_counts[label] += 1


# Close files
for f in uint_digit_files.values():
    f.close()

for f in int_digit_files.values():
    f.close()

for f in bin_digit_files.values():
    f.close()

# Just some statistics
for d in range(num_classes):
    print(f"Digit {d}: Collected {digit_counts[d]} samples.")

total_samples = sum(digit_counts.values())
print(f"Total samples collected: {total_samples}")
