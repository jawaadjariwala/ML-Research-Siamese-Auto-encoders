"""Data loading and preprocessing module for NSL-KDD dataset."""

import os
from pathlib import Path
from typing import Tuple, Optional
import urllib.request
import zipfile
import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class NSLKDDDataset(Dataset):
    """PyTorch Dataset for NSL-KDD network intrusion detection data.
    
    This dataset extends PyTorch's Dataset class to handle network traffic
    features for anomaly detection. It supports both normal and anomalous samples.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
    ) -> None:
        """Initialize NSL-KDD dataset.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Optional label array. If None, dataset is unlabeled.
            transform: Optional transform to apply to samples
        """
        self.features = torch.FloatTensor(features)
        self.labels = labels if labels is None else torch.LongTensor(labels)
        self.transform = transform

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label) if labels exist, else (features,)
        """
        sample = self.features[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            return sample, self.labels[idx]
        return (sample,)


def download_nsl_kdd_data(data_dir: str = "data") -> str:
    """Download NSL-KDD dataset if not already present.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Path to the dataset directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    train_file = data_path / "KDDTrain+.txt"
    test_file = data_path / "KDDTest+.txt"
    
    # Check if files already exist
    if train_file.exists() and test_file.exists():
        print(f"NSL-KDD dataset already exists at {data_path}")
        return str(data_path)
    
    print("Downloading NSL-KDD dataset...")
    print("This may take a few minutes...")
    
    # NSL-KDD dataset URLs (using direct links from UNB)
    base_url = "https://www.unb.ca/cic/datasets/nsl.html"
    
    # Alternative: Use Kaggle or other sources
    # For now, we'll provide instructions and try common locations
    print("\n" + "="*60)
    print("NSL-KDD Dataset Download Required")
    print("="*60)
    print("\nPlease download the NSL-KDD dataset from:")
    print("  Official: https://www.unb.ca/cic/datasets/nsl.html")
    print("\nOr use one of these alternative sources:")
    print("  - Kaggle: https://www.kaggle.com/datasets/hassan06/nslkdd")
    print("  - GitHub: Various repositories host the dataset")
    print("\nRequired files:")
    print("  - KDDTrain+.txt")
    print("  - KDDTest+.txt")
    print(f"\nPlace these files in: {data_path.absolute()}")
    print("="*60 + "\n")
    
    # Try to download from a public mirror if available
    try:
        # Common mirror locations (these may change)
        mirrors = [
            "https://github.com/defcom17/NSL_KDD/raw/master/KDDTrain%2B.txt",
            "https://github.com/defcom17/NSL_KDD/raw/master/KDDTest%2B.txt",
        ]
        
        print("Attempting to download from GitHub mirror...")
        urllib.request.urlretrieve(mirrors[0], train_file)
        print(f"✓ Downloaded {train_file.name}")
        
        urllib.request.urlretrieve(mirrors[1], test_file)
        print(f"✓ Downloaded {test_file.name}")
        
        print(f"\n✓ Dataset downloaded successfully to {data_path.absolute()}")
        return str(data_path)
        
    except Exception as e:
        print(f"\n⚠ Automatic download failed: {e}")
        print("\nPlease download manually:")
        print(f"1. Create directory: {data_path.absolute()}")
        print("2. Download KDDTrain+.txt and KDDTest+.txt")
        print("3. Place files in the directory above")
        print("\nThe dataset will be loaded automatically once files are in place.")
        
        # Create directory structure
        data_path.mkdir(exist_ok=True)
        return str(data_path)


def load_nsl_kdd_data(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess NSL-KDD dataset.
    
    This function handles loading the NSL-KDD dataset, encoding categorical
    features, scaling numerical features, and splitting into train/val/test sets.
    
    Args:
        data_path: Path to NSL-KDD data directory. If None, looks for data in
            standard locations or generates synthetic data for demonstration.
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test) as numpy arrays
    """
    np.random.seed(random_state)
    
    # Auto-download if data_path is None
    if data_path is None:
        data_path = download_nsl_kdd_data()
    
    # Load actual NSL-KDD data
    data_path_obj = Path(data_path)
    train_file = data_path_obj / "KDDTrain+.txt"
    test_file = data_path_obj / "KDDTest+.txt"
    
    if not train_file.exists():
        # Try to download
        data_path = download_nsl_kdd_data()
        data_path_obj = Path(data_path)
        train_file = data_path_obj / "KDDTrain+.txt"
        test_file = data_path_obj / "KDDTest+.txt"
    
    if train_file.exists():
        print(f"Loading NSL-KDD dataset from {data_path_obj.absolute()}")
        
        # Read CSV files (NSL-KDD uses comma separation)
        print("Reading training data...")
        train_df = pd.read_csv(train_file, header=None, low_memory=False)
        
        print("Reading test data...")
        test_df = pd.read_csv(test_file, header=None, low_memory=False) if test_file.exists() else None
        
        # NSL-KDD has 41 features + 1 label (last column)
        # Features at indices 1, 2, 3 are categorical (protocol_type, service, flag)
        # Column 41 (index 40) is the label
        
        print("Preprocessing data...")
        
        # Check number of columns
        n_cols_train = train_df.shape[1]
        n_cols_test = test_df.shape[1] if test_df is not None else n_cols_train
        print(f"Columns in training data: {n_cols_train}")
        print(f"Columns in test data: {n_cols_test}")
        
        # NSL-KDD structure:
        # - Columns 0-40: 41 features
        # - Column 41 (index 41): Attack type label (normal, neptune, back, etc.)
        # - Column 42 (index 42): Difficulty level (optional, may not be present)
        
        # Extract labels from column 41 (0-indexed), not the last column
        label_col_idx = 41
        if n_cols_train > label_col_idx:
            y_train_raw = train_df.iloc[:, label_col_idx].values.astype(str)
            y_train = np.array([label.strip() for label in y_train_raw])  # Strip whitespace
        else:
            # Fallback: use last column if column 41 doesn't exist
            y_train_raw = train_df.iloc[:, -1].values.astype(str)
            y_train = np.array([label.strip() for label in y_train_raw])
        
        if test_df is not None:
            if n_cols_test > label_col_idx:
                y_test_raw = test_df.iloc[:, label_col_idx].values.astype(str)
                y_test = np.array([label.strip() for label in y_test_raw])
            else:
                y_test_raw = test_df.iloc[:, -1].values.astype(str)
                y_test = np.array([label.strip() for label in y_test_raw])
        else:
            y_test = None
        
        # Debug: Show unique labels
        unique_labels = np.unique(y_train)
        print(f"Unique labels in training data: {unique_labels[:10]}...")  # Show first 10
        normal_count = np.sum([label.lower() == 'normal' for label in y_train])
        print(f"Count of 'normal' labels: {normal_count} out of {len(y_train)}")
        
        # Encode labels: normal = 0, anomaly = 1
        # Use direct string comparison (case-insensitive) instead of LabelEncoder
        y_train = np.array([0 if label.lower() == 'normal' else 1 for label in y_train], dtype=int)
        
        if y_test is not None:
            y_test = np.array([0 if label.lower() == 'normal' else 1 for label in y_test], dtype=int)
        
        # Extract features - NSL-KDD should have exactly 41 features
        # Handle case where there might be extra columns (like index or trailing comma)
        if n_cols_train > 42:
            # If more than 42 columns, take first 41 (features)
            X_train = train_df.iloc[:, :41].copy()
            print(f"⚠️  Warning: {n_cols_train} columns detected, using first 41 as features")
        elif n_cols_train == 42:
            # Standard case: 41 features + 1 label
            X_train = train_df.iloc[:, :41].copy()  # Take first 41 columns (0-40)
        else:
            # Fallback: take all except last (should be 41 features)
            X_train = train_df.iloc[:, :-1].copy()
            if X_train.shape[1] != 41:
                print(f"⚠️  Warning: Expected 41 features, got {X_train.shape[1]}. Using first 41.")
                X_train = train_df.iloc[:, :41].copy()
        
        if test_df is not None:
            if n_cols_test > 42:
                X_test = test_df.iloc[:, :41].copy()
            elif n_cols_test == 42:
                X_test = test_df.iloc[:, :41].copy()
            else:
                X_test = test_df.iloc[:, :-1].copy()
                if X_test.shape[1] != 41:
                    X_test = test_df.iloc[:, :41].copy()
        else:
            X_test = None
        
        print(f"Feature matrix shape: {X_train.shape} (should be (N, 41))")
        
        # Handle categorical features (columns at indices 1, 2, 3)
        # These are: protocol_type, service, flag
        categorical_cols = [1, 2, 3]
        label_encoders = {}
        
        for col_idx in categorical_cols:
            le_feat = LabelEncoder()
            # Fit on training data
            X_train.iloc[:, col_idx] = le_feat.fit_transform(X_train.iloc[:, col_idx].astype(str))
            label_encoders[col_idx] = le_feat
            
            # Transform test data using same encoder
            if X_test is not None:
                # Handle unseen categories in test set
                test_cats = X_test.iloc[:, col_idx].astype(str).values
                known_cats = set(le_feat.classes_)
                # Replace unknown categories with most common category
                most_common = le_feat.classes_[0]
                test_cats = [cat if cat in known_cats else most_common for cat in test_cats]
                X_test.iloc[:, col_idx] = le_feat.transform(test_cats)
        
        # Convert to numpy arrays and handle numeric conversion
        print("Converting to numeric format...")
        X_train = X_train.values
        X_test = X_test.values if X_test is not None else None
        
        # Convert all to float, handling any remaining non-numeric values
        for i in range(X_train.shape[1]):
            if i not in categorical_cols:  # Skip already encoded categorical columns
                X_train[:, i] = pd.to_numeric(X_train[:, i], errors='coerce').astype(float)
        
        if X_test is not None:
            for i in range(X_test.shape[1]):
                if i not in categorical_cols:
                    X_test[:, i] = pd.to_numeric(X_test[:, i], errors='coerce').astype(float)
        
        # Ensure exactly 41 features
        if X_train.shape[1] != 41:
            print(f"⚠️  Adjusting feature count from {X_train.shape[1]} to 41")
            X_train = X_train[:, :41]
        if X_test is not None and X_test.shape[1] != 41:
            X_test = X_test[:, :41]
        
        # Combine train and test for unified preprocessing
        if X_test is not None:
            X = np.vstack([X_train, X_test])
            y = np.hstack([y_train, y_test])
        else:
            X = X_train
            y = y_train
        
        # Final verification
        if X.shape[1] != 41:
            print(f"⚠️  WARNING: Final feature count is {X.shape[1]}, trimming to 41")
            X = X[:, :41]
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"  Normal samples: {np.sum(y == 0)} ({100*np.mean(y == 0):.1f}%)")
        print(f"  Anomaly samples: {np.sum(y == 1)} ({100*np.mean(y == 1):.1f}%)")
        
    else:
        raise FileNotFoundError(
            f"NSL-KDD data not found at {data_path}. "
            f"Please download KDDTrain+.txt and KDDTest+.txt and place them in {data_path_obj.absolute()}"
        )
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_loaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = NSLKDDDataset(X_train, y_train)
    val_dataset = NSLKDDDataset(X_val, y_val)
    test_dataset = NSLKDDDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

