"""
一键将 FRED 数据集的 train/test 两个 split 转换为 COCO 标注。

本脚本是 convert_fred_to_coco.py 的批量入口，推荐日常直接使用它重新生成
coco_annotations/train.json 和 coco_annotations/test.json。

示例：
python tools/convert_all_fred_to_coco.py \
  --fred-root F:\FRED \
  --output-dir F:\FRED\coco_annotations \
  --image-width 1280 \
  --image-height 720

服务器示例：
python tools/convert_all_fred_to_coco.py \
  --fred-root /mnt/data/cc/FRED \
  --output-dir /mnt/data/cc/FRED/coco_annotations \
  --image-width 1280 \
  --image-height 720
"""
import subprocess
import os
import argparse
from pathlib import Path


def convert_all_splits(fred_root, output_dir, timestamp_scale=1000000, image_width=None, image_height=None):
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
            "--split", split,
            "--timestamp-scale", str(timestamp_scale),
        ]
        if image_width is not None and image_height is not None:
            cmd.extend(["--image-width", str(image_width), "--image-height", str(image_height)])
        
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
    parser = argparse.ArgumentParser(description="Convert all FRED splits to COCO format")
    parser.add_argument("--fred-root", type=str, default="/mnt/data/cc/FRED",
                        help="Root directory of FRED dataset")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for COCO json files. Defaults to <fred-root>/coco_annotations")
    parser.add_argument("--timestamp-scale", type=int, default=1000000,
                        help="Scale used to map coordinates.txt timestamps to frame filename timestamps")
    parser.add_argument("--image-width", type=int, default=None,
                        help="Optional fixed image width")
    parser.add_argument("--image-height", type=int, default=None,
                        help="Optional fixed image height")
    args = parser.parse_args()
    
    output_dir = args.output_dir or str(Path(args.fred_root) / "coco_annotations")
    convert_all_splits(
        args.fred_root,
        output_dir,
        timestamp_scale=args.timestamp_scale,
        image_width=args.image_width,
        image_height=args.image_height,
    )
