"""
Script to convert all FRED dataset splits to COCO format
"""
import subprocess
import os
from pathlib import Path


def convert_all_splits(fred_root, output_dir):
    """Convert both train and test splits of FRED dataset to COCO format."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define splits
    splits = ["train", "test"]
    
    for split in splits:
        print(f"Converting {split} split...")
        
        # Prepare arguments for the conversion script
        cmd = [
            "python", "convert_fred_to_coco.py",
            "--fred-root", fred_root,
            "--output-dir", str(output_dir),
            "--split", split
        ]
        
        # Run the conversion script
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
        
        if result.returncode != 0:
            print(f"Error converting {split} split!")
            return False
        else:
            print(f"Successfully converted {split} split.")
    
    print("All splits converted successfully!")
    return True


if __name__ == "__main__":
    # Define paths
    FRED_ROOT = "/mnt/data/cc/FRED"
    OUTPUT_DIR = "/mnt/data/cc/FRED/coco_annotations"
    
    convert_all_splits(FRED_ROOT, OUTPUT_DIR)