"""
setup_datasets.py  -  Helper script to verify and structure dataset files after extraction.

Expected Structure:

datasets/
├── primary/                  (Dataset 4: Kaggle License Plate Detection)
│   ├── images/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── valid/
│       └── test/
│
└── supplement/               (Dataset 1: Indian Vehicle Dataset)
    ├── State-wise_OLX/       (or XML + image files directly inside supplement/)
    └── ...
"""

import os
import glob
import shutil

PRIMARY_DIR = os.path.join("datasets", "primary")
SUPPLEMENT_DIR = os.path.join("datasets", "supplement")

def check_structure():
    print("=== Checking Dataset Directory Structure ===")
    
    # Check Primary
    primary_images = glob.glob(os.path.join(PRIMARY_DIR, "**", "*.jpg"), recursive=True) + \
                     glob.glob(os.path.join(PRIMARY_DIR, "**", "*.png"), recursive=True)
    primary_labels = glob.glob(os.path.join(PRIMARY_DIR, "**", "*.txt"), recursive=True)
    
    print(f"\n[Primary Dataset (Dataset 4)]")
    print(f"  Target path: {PRIMARY_DIR}")
    print(f"  Images found: {len(primary_images)}")
    print(f"  Labels found: {len(primary_labels)}")
    
    # Check Supplement
    supp_images = glob.glob(os.path.join(SUPPLEMENT_DIR, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(SUPPLEMENT_DIR, "**", "*.png"), recursive=True)
    supp_labels = glob.glob(os.path.join(SUPPLEMENT_DIR, "**", "*.xml"), recursive=True)
    
    print(f"\n[Supplement Dataset (Dataset 1)]")
    print(f"  Target path: {SUPPLEMENT_DIR}")
    print(f"  Images found: {len(supp_images)}")
    print(f"  Labels found: {len(supp_labels)}")
    
    if len(primary_images) > 0 or len(supp_images) > 0:
        print("\n[OK] Datasets detected! You can run:")
        print("   python evaluate.py --max 50")
    else:
        print("\n[INFO] No dataset images detected yet.")
        print("   Please download the datasets from Kaggle and place/extract them into:")
        print(f"   1. Primary (Dataset 4): {os.path.abspath(PRIMARY_DIR)}")
        print(f"   2. Supplement (Dataset 1): {os.path.abspath(SUPPLEMENT_DIR)}")

if __name__ == "__main__":
    check_structure()
