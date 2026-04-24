"""
Complete script to prepare and run MvHeatDET training on FRED dataset
"""
import os
import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, cwd=None):
    """Run a command and check if it succeeded."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        sys.exit(result.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare and run MvHeatDET training on FRED dataset")
    parser.add_argument("--prepare-data", action="store_true", help="Convert FRED dataset to COCO format")
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--test", action="store_true", help="Start testing")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for testing")
    parser.add_argument("--data-path", type=str, default="/mnt/data/cc/FRED", help="Path to FRED dataset")
    
    args = parser.parse_args()
    
    # Paths
    tools_dir = Path(__file__).parent / "tools"
    data_path = Path(args.data_path)
    coco_annotations_path = data_path / "coco_annotations"
    
    # Create annotations directory if needed
    coco_annotations_path.mkdir(exist_ok=True)
    
    # Step 1: Prepare data if requested
    if args.prepare_data:
        print("Step 1: Converting FRED dataset to COCO format...")
        
        # Run the conversion script
        cmd = ["python", "convert_all_fred_to_coco.py"]
        run_command(cmd, cwd=tools_dir)
    
    # Verify that annotations exist
    train_ann = coco_annotations_path / "train.json"
    test_ann = coco_annotations_path / "test.json"
    
    if not train_ann.exists() or not test_ann.exists():
        print(f"Annotation files not found at {coco_annotations_path}")
        print("Please run with --prepare-data flag or create annotation files manually.")
        sys.exit(1)
    
    print(f"Found annotation files: {train_ann}, {test_ann}")
    
    # Step 2: Run training if requested
    if args.train:
        print("Step 2: Starting training on FRED dataset...")
        
        # Command for single GPU training
        cmd = [
            "python", "train.py", 
            "-c", "../configs/fred_complete.yml",
            "--amp"
        ]
        
        run_command(cmd, cwd=tools_dir)
    
    # Step 3: Run testing if requested
    if args.test:
        if not args.checkpoint:
            print("Please provide a checkpoint path for testing using --checkpoint")
            sys.exit(1)
        
        print("Step 3: Starting testing on FRED dataset...")
        
        # Command for testing
        cmd = [
            "python", "train.py", 
            "-c", "../configs/fred_complete.yml",
            "-r", args.checkpoint,
            "--test-only"
        ]
        
        run_command(cmd, cwd=tools_dir)


if __name__ == "__main__":
    main()