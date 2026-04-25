#!/usr/bin/env python3
"""Script to download NSL-KDD dataset."""

import os
import urllib.request
from pathlib import Path


def download_nsl_kdd():
    """Download NSL-KDD dataset files."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    train_file = data_dir / "KDDTrain+.txt"
    test_file = data_dir / "KDDTest+.txt"
    
    if train_file.exists() and test_file.exists():
        print("✓ NSL-KDD dataset already downloaded")
        return
    
    print("Downloading NSL-KDD dataset...")
    print("Source: GitHub mirror (defcom17/NSL_KDD)")
    
    urls = {
        "KDDTrain+.txt": "https://github.com/defcom17/NSL_KDD/raw/master/KDDTrain%2B.txt",
        "KDDTest+.txt": "https://github.com/defcom17/NSL_KDD/raw/master/KDDTest%2B.txt",
    }
    
    for filename, url in urls.items():
        filepath = data_dir / filename
        if filepath.exists():
            print(f"✓ {filename} already exists")
            continue
        
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            print(f"\nPlease download manually from:")
            print(f"  Official: https://www.unb.ca/cic/datasets/nsl.html")
            print(f"  Kaggle: https://www.kaggle.com/datasets/hassan06/nslkdd")
            print(f"\nPlace {filename} in: {data_dir.absolute()}")
            return
    
    print(f"\n✓ Dataset downloaded successfully to {data_dir.absolute()}")
    print(f"  Files: {train_file.name}, {test_file.name}")


if __name__ == "__main__":
    download_nsl_kdd()

