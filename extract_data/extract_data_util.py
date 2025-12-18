# --------------------------------------------
# Copyright 2024 KU Leuven
# Ryan Antonio <ryan.antonio@esat.kuleuven.be>

# Description:
# Utility programs and functions data extraction
# --------------------------------------------

import urllib.request
import zipfile
import tarfile
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

    print("Done.")

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