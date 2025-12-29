"""
Utility functions for data extraction and processing.
"""

import urllib.request
import zipfile
import tarfile
import random
from tqdm import tqdm
from pathlib import Path

# For downloading and extracting archives
def download_and_extract(
    url,
    out_dir="data",
    filename=None,
    delete_archive=False,
):
    """
    Download an archive from a URL and extract it.

    Args:
        url (str): Download URL
        out_dir (str | Path): Output directory
        filename (str | None): Optional override for downloaded filename
        delete_archive (bool): Delete archive after successful extraction
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1]

    archive_path = out_dir / filename

    # Download (skip if already exists)
    if not archive_path.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, archive_path)
    else:
        print(f"{filename} already exists, skipping download.")

    # Extract
    print("Extracting...")
    try:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(out_dir)
        elif filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(out_dir)
        else:
            raise ValueError(f"Unsupported archive format: {filename}")
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}")

    # Optional cleanup
    if delete_archive:
        archive_path.unlink()
        print(f"Deleted archive: {archive_path.name}")

    print("Extraction complete!")

def read_data(class_list, data_path):
    """
    Read data from files for each class label.
    Args:
        class_list: list of class labels
        data_path: path to the data files
    Returns:
        X_data: dictionary with class labels as keys and data lists as values
    """
    X_data = dict()
    for class_label in tqdm(class_list, desc="Reading data"):
        # Training dataset
        read_file = f"{data_path}/{class_label}.txt"
        X_data[class_label] = load_dataset(read_file)
    return X_data


def split_data(X_data, class_list, split_percent = 0.8):
    """
    Split data into two parts based on split_percent.
    Args:
        X_data: dictionary with class labels as keys and data lists as values
        class_list: list of class labels
        split_percent: percentage of data to go into first split
    Returns:
        X_split_data1: first split of data
        X_split_data2: second split of data
    """
    # Initialize empty dictionaries
    X_split_data1 = dict()
    X_split_data2 = dict()

    for class_label in tqdm(class_list, desc="Splitting data"):
        # Get item counts first
        item_len = len(X_data[class_label])
        split1_len = round(item_len * split_percent)
        split2_len = item_len - split1_len

        # Randomize the contents of the list first
        random.shuffle(X_data[class_label])

        # Get split 1 first
        split1_list = []
        for item_num in range(split1_len):
            split1_list.append(X_data[class_label][item_num])

        # Get split 2 next but starts at split1_len count
        split2_list = []
        for item_num in range(split1_len,split1_len+split2_len):
            split2_list.append(X_data[class_label][item_num])

        # Load into dictionaries
        X_split_data1[class_label] = split1_list
        X_split_data2[class_label] = split2_list

    return X_split_data1, X_split_data2

# For reading and loading a data
def load_dataset(file_path):
    """
    Load dataset from a text file.
    Args:
        file_path (str): Path to the text file
    Returns:
        dataset (list): Loaded dataset as a list of lists
    """
    # Initialize empty data set array
    dataset = []
    with open(file_path, "r") as rf:
        for line in rf:
            line = line.strip().split()
            int_line = [int(x) for x in line]
            dataset.append(int_line)
    # Close the file
    rf.close()
    return dataset


# For scaling values to 0-255
def scale_to_255(value):
    """
    Conversion from [-1, 1] to [0, 255]

    Args:
        value (float): Input value within [-1, 1]
    Returns:
        scaled (int): Scaled value within [0, 255]
    """
    if value < -1 or value > 1:
        raise ValueError("Input must be within [-1, 1]")
    scaled = int(((value + 1) / 2) * 255)
    return scaled


# For quantizing the values through a binning process
def binning_quantization(value, bin_size, num_bins):
    """
    Quantize a value into bins.

    Args:
        value (int): Input value
        bin_size (int): Size of each bin
        num_bins (int): Number of bins
    Returns:
        quantized_value (int): Quantized bin index
    """

    quantized_value = value // bin_size
    if quantized_value >= num_bins:
        quantized_value = num_bins - 1
    return quantized_value


# Convert levels of a dataset
def convert_levels(dataset, max_val, val_levels):
    """
    Convert levels of a dataset to different levels.
    Args:
        dataset: input dataset (dict of lists)
        max_val: maximum possible value
        val_levels: destination levels
    Returns:
        converted dataset (dict of lists)
    """
    dst_levels = (max_val // val_levels)
    dataset_copy = dict()
    for key in dataset:
        for j in range(len(dataset[key])):
            if key not in dataset_copy:
                dataset_copy[key] = []
            dataset_copy[key].append([
                (x//dst_levels) for x in dataset[key][j]
            ])
    return dataset_copy


# Binning quantization
def convert_binning_quantization(dataset, max_val, num_bins):
    """
    Quantize a value into bins.

    Args:
        dataset: input dataset (dict of lists)
        max_val (int): Maximum possible value
        num_bins (int): Number of bins

    Returns:
        quantized_value (int): Quantized bin index
    """
    # For determining bin size
    bin_size = max_val // num_bins
    dataset_copy = dict()
    for key in dataset:
        for j in range(len(dataset[key])):

            # Initialize dictionary key
            if key not in dataset_copy:
                dataset_copy[key] = []

            dataset_copy[key].append([
                binning_quantization(x, bin_size, num_bins) for x in dataset[key][j]
            ])

    return dataset_copy
