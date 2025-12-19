# --------------------------------------------
# Copyright 2024 KU Leuven
# Ryan Antonio <ryan.antonio@esat.kuleuven.be>

# Description:
# Utility programs and functions data extraction
# --------------------------------------------

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

# Read data
def read_data(class_list, data_path):
    X_data = dict()
    for class_label in tqdm(class_list, desc="Reading data"):
        # Training dataset
        read_file = f"{data_path}/{class_label}.txt"
        X_data[class_label] = load_dataset(read_file)
    return X_data
        
# For splitting data into two sets
def split_data(X_data, class_list, split_percent = 0.8):
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
    """
    if value < -1 or value > 1:
        raise ValueError("Input must be within [-1, 1]")
    scaled = int(((value + 1) / 2) * 255)
    return scaled