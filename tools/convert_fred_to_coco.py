"""
将 FRED 数据集的单个 split 转换为 MvHeatDET 训练所需的 COCO 标注。

本脚本会从事件帧文件名末尾解析真实时间戳，再与 coordinates.txt 中的
时间戳匹配，避免按帧序号估算时间导致图像和标注错位。

示例：
python tools/convert_fred_to_coco.py \
  --fred-root F:\FRED \
  --output-dir F:\FRED\coco_annotations \
  --split train \
  --image-width 1280 \
  --image-height 720

服务器示例：
python tools/convert_fred_to_coco.py \
  --fred-root /mnt/data/cc/FRED \
  --output-dir /mnt/data/cc/FRED/coco_annotations \
  --split train \
  --image-width 1280 \
  --image-height 720
"""
import os
import json
import cv2
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from glob import glob


def parse_fred_annotations(annotation_file, timestamp_scale=1000000):
    """Parse FRED annotation file and return a dictionary mapping timestamps to annotations."""
    annotations = {}
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse line: timestamp: x1, y1, x2, y2, id, class_name
            parts = line.split(': ')
            if len(parts) != 2:
                continue
            timestamp = parts[0]
            bbox_parts = parts[1].split(', ')
            
            if len(bbox_parts) >= 6:
                x1, y1, x2, y2, obj_id, class_name = bbox_parts[0], bbox_parts[1], bbox_parts[2], bbox_parts[3], bbox_parts[4], ', '.join(bbox_parts[5:])
                timestamp_key = int(round(float(timestamp) * timestamp_scale))
                
                if timestamp_key not in annotations:
                    annotations[timestamp_key] = []
                    
                annotations[timestamp_key].append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'id': int(float(obj_id)),
                    'class_name': class_name
                })
    return annotations


def frame_timestamp_key(frame_path):
    """Extract the trailing integer timestamp from FRED frame names."""
    match = re.search(r'_(\d+)\.png$', os.path.basename(frame_path))
    if match is None:
        return None
    return int(match.group(1))


def clamp_bbox_xyxy(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def convert_fred_to_coco(
    fred_root,
    output_dir,
    split='train',
    timestamp_scale=1000000,
    image_width=None,
    image_height=None,
):
    """
    Convert FRED dataset to COCO format.
    
    Args:
        fred_root: Root directory of FRED dataset
        output_dir: Output directory for COCO format dataset
        split: 'train' or 'test'
        timestamp_scale: scale used by frame filename timestamps and coordinates.txt
        image_width: optional fixed image width to avoid reading every frame
        image_height: optional fixed image height to avoid reading every frame
    """
    print(f"Converting FRED {split} dataset to COCO format...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize COCO format structure
    coco_format = {
        "info": {
            "description": f"FRED {split} dataset converted to COCO format",
            "version": "1.0",
            "year": 2025,
            "contributor": "FRED Dataset Contributors",
            "date_created": "2025"
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "drone", "supercategory": "object"}],
        "images": [],
        "annotations": []
    }
    
    # Get all sequences in the split
    sequences = sorted(glob(os.path.join(fred_root, split, '*')))
    img_id = 0
    ann_id = 0
    
    # Iterate through each sequence
    for seq_path in tqdm(sequences):
        if not os.path.isdir(seq_path):
            continue
            
        seq_name = os.path.basename(seq_path)
        
        # Get annotation file
        annotation_file = os.path.join(seq_path, 'coordinates.txt')
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found for sequence {seq_path}")
            continue
            
        # Parse annotations
        annotations_by_time = parse_fred_annotations(annotation_file, timestamp_scale)
        
        # Get event frames
        event_frame_dir = os.path.join(seq_path, 'Event', 'Frames')
        if not os.path.exists(event_frame_dir):
            print(f"Warning: Event frames not found for sequence {seq_path}")
            continue
            
        event_frames = sorted(
            glob(os.path.join(event_frame_dir, '*.png')),
            key=lambda p: frame_timestamp_key(p) if frame_timestamp_key(p) is not None else os.path.basename(p)
        )
        
        # Process each frame
        for frame_path in event_frames:
            frame_filename = os.path.basename(frame_path)
            timestamp_key = frame_timestamp_key(frame_path)
            if timestamp_key is None:
                print(f"Warning: Could not parse timestamp from frame name {frame_path}")
                continue
            
            # Read image to get dimensions
            if image_width is not None and image_height is not None:
                width, height = image_width, image_height
            else:
                img = cv2.imread(frame_path)
                if img is None:
                    print(f"Warning: Could not read image {frame_path}")
                    continue
                height, width = img.shape[:2]
            
            # Add image info to COCO format
            coco_format["images"].append({
                "id": img_id,
                "file_name": f"{split}/{seq_name}/Event/Frames/{frame_filename}",
                "width": width,
                "height": height,
                "seq_name": seq_name
            })
            
            # Add annotations for this frame if they exist
            if timestamp_key in annotations_by_time:
                for ann in annotations_by_time[timestamp_key]:
                    clamped = clamp_bbox_xyxy(ann['bbox'], width, height)
                    if clamped is None:
                        continue
                    x1, y1, x2, y2 = clamped
                    
                    # Convert x2, y2 to width, height (COCO format)
                    width_bbox = x2 - x1
                    height_bbox = y2 - y1
                    
                    coco_format["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,  # Only drone class
                        "bbox": [x1, y1, width_bbox, height_bbox],
                        "area": width_bbox * height_bbox,
                        "iscrowd": 0
                    })
                    ann_id += 1
            
            img_id += 1
    
    # Save the COCO format JSON
    output_json_path = output_dir / f"{split}.json"
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f)
    
    print(f"Conversion completed. Saved to {output_json_path}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")


def main():
    parser = argparse.ArgumentParser(description="Convert FRED dataset to COCO format")
    parser.add_argument("--fred-root", type=str, required=True, 
                        help="Root directory of FRED dataset")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Output directory for COCO format dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="Dataset split to convert")
    parser.add_argument("--timestamp-scale", type=int, default=1000000,
                        help="Scale used to map coordinates.txt timestamps to frame filename timestamps")
    parser.add_argument("--image-width", type=int, default=None,
                        help="Optional fixed image width")
    parser.add_argument("--image-height", type=int, default=None,
                        help="Optional fixed image height")
    
    args = parser.parse_args()
    
    convert_fred_to_coco(
        args.fred_root,
        args.output_dir,
        args.split,
        timestamp_scale=args.timestamp_scale,
        image_width=args.image_width,
        image_height=args.image_height,
    )


if __name__ == "__main__":
    main()
