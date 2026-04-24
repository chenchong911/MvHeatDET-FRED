"""
Script to convert FRED dataset to COCO format for MvHeatDET
"""
import os
import json
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from glob import glob


def parse_fred_annotations(annotation_file):
    """Parse FRED annotation file and return a dictionary mapping timestamps to annotations."""
    annotations = {}
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse line: timestamp: x1, y1, x2, y2, id, class_name
            parts = line.split(': ')
            timestamp = parts[0]
            bbox_parts = parts[1].split(', ')
            
            if len(bbox_parts) >= 6:
                x1, y1, x2, y2, obj_id, class_name = bbox_parts[0], bbox_parts[1], bbox_parts[2], bbox_parts[3], bbox_parts[4], ', '.join(bbox_parts[5:])
                
                if timestamp not in annotations:
                    annotations[timestamp] = []
                    
                annotations[timestamp].append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'id': int(float(obj_id)),
                    'class_name': class_name
                })
    return annotations


def convert_fred_to_coco(fred_root, output_dir, split='train'):
    """
    Convert FRED dataset to COCO format.
    
    Args:
        fred_root: Root directory of FRED dataset
        output_dir: Output directory for COCO format dataset
        split: 'train' or 'test'
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
        annotations_by_time = parse_fred_annotations(annotation_file)
        
        # Get event frames
        event_frame_dir = os.path.join(seq_path, 'Event', 'Frames')
        if not os.path.exists(event_frame_dir):
            print(f"Warning: Event frames not found for sequence {seq_path}")
            continue
            
        event_frames = sorted(glob(os.path.join(event_frame_dir, '*.png')))
        
        # Process each frame
        for frame_idx, frame_path in enumerate(event_frames):
            frame_filename = os.path.basename(frame_path)
            
            # Try to get timestamp for this frame
            # Assuming frames are indexed sequentially starting from 1
            timestamp = f"{float((frame_idx+1)*0.033333):.6f}"
            
            # Read image to get dimensions
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
            if timestamp in annotations_by_time:
                for ann in annotations_by_time[timestamp]:
                    x1, y1, x2, y2 = ann['bbox']
                    
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
    
    args = parser.parse_args()
    
    convert_fred_to_coco(args.fred_root, args.output_dir, args.split)


if __name__ == "__main__":
    main()