#!/usr/bin/env python3
"""Quick test to verify NSL-KDD data files are present."""

from pathlib import Path

print("Checking NSL-KDD dataset files...")
print("=" * 60)

data_dir = Path("data")
train_file = data_dir / "KDDTrain+.txt"
test_file = data_dir / "KDDTest+.txt"

if train_file.exists() and test_file.exists():
    print(f"✓ Dataset files found in {data_dir.absolute()}")
    print(f"  - {train_file.name}: {train_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  - {test_file.name}: {test_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Quick check of file format
    print("\nChecking file format...")
    with open(train_file, 'r') as f:
        first_line = f.readline().strip()
        num_cols = len(first_line.split(','))
        print(f"  First line columns: {num_cols} (expected 42: 41 features + 1 label)")
    
    print("\n✓ Dataset is ready!")
    print("\nNote: To test full data loading, install dependencies first:")
    print("  poetry install")
    print("  poetry run python -c 'from finalproject.data import load_nsl_kdd_data; load_nsl_kdd_data(\"data\")'")
else:
    print(f"✗ Dataset files not found in {data_dir.absolute()}")
    print("\nPlease run: python download_data.py")

