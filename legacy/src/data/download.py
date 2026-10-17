"""
Download and extract the ROCOv2 dataset from Zenodo.

ROCOv2 (Radiology Objects in COntext, version 2) is distributed under
CC BY-NC 4.0 (research / non-commercial use only). See:
https://zenodo.org/records/10821435
"""

import os
import subprocess

ZENODO_ARCHIVE_URL = "https://zenodo.org/api/records/10821435/files-archive"


def download_and_extract(data_root: str = "./data") -> None:
    """Download the ROCOv2 archive and unzip train/test image sets.

    Mirrors the manual steps used in the original notebook, so this can be
    run either as a script or imported and called from a training pipeline.
    """
    train_dir = os.path.join(data_root, "train_ROCOv2")
    test_dir = os.path.join(data_root, "test_ROCOv2")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    archive_path = os.path.join(data_root, "ROCOV2.zip")

    print("Downloading ROCOv2 archive from Zenodo...")
    subprocess.run(["wget", "-O", archive_path, ZENODO_ARCHIVE_URL], check=True)

    print("Extracting main archive...")
    subprocess.run(["unzip", "-o", archive_path, "-d", data_root], check=True)

    print("Extracting train images...")
    subprocess.run(
        ["unzip", "-o", os.path.join(data_root, "train_images.zip"), "-d", train_dir],
        check=True,
    )

    print("Extracting test images...")
    subprocess.run(
        ["unzip", "-o", os.path.join(data_root, "test_images.zip"), "-d", test_dir],
        check=True,
    )

    print("Done. Data is ready under:", data_root)


if __name__ == "__main__":
    download_and_extract()
