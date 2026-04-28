"""
可视化 COCO 标注是否和 FRED 事件帧对齐。

示例：只检查 train split 的第 0 个序列，输出前 200 张带框图片。

python tools\visualize_coco_annotations.py --coco-ann F:\FRED\coco_annotations\test.json --fred-root F:\FRED --split test --seq-name 8 --output-dir F:\FRED\visualizations\train_183 --max-images 50 --only-with-annotations

Linux 示例：

python tools/visualize_coco_annotations.py \
  --coco-ann /mnt/data/cc/FRED/coco_annotations/train.json \
  --fred-root /mnt/data/cc/FRED \
  --seq-name 0 \
  --output-dir /mnt/data/cc/FRED/visualizations/train_0 \
  --max-images 200
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2


def load_coco(path):
    """Load a standard COCO JSON file, with a small fallback for JSONL records."""
    path = Path(path)
    if path.suffix.lower() != ".jsonl":
        return json.loads(path.read_text(encoding="utf-8"))

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) == 1 and isinstance(records[0], dict) and "images" in records[0]:
        return records[0]

    coco = {"images": [], "annotations": [], "categories": []}
    for record in records:
        record_type = record.get("type")
        if record_type == "image":
            coco["images"].append(record)
        elif record_type == "annotation":
            coco["annotations"].append(record)
        elif record_type == "category":
            coco["categories"].append(record)
    return coco


def normalize_rel_path(path):
    return str(path).replace("\\", "/")


def image_matches(image_info, split=None, seq_name=None):
    file_name = normalize_rel_path(image_info.get("file_name", ""))
    parts = file_name.split("/")

    if split is not None and (not parts or parts[0] != split):
        return False

    if seq_name is not None:
        info_seq = image_info.get("seq_name")
        if info_seq is not None:
            return str(info_seq) == str(seq_name)
        return len(parts) > 1 and parts[1] == str(seq_name)

    return True


def draw_label(image, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 4)
    cv2.rectangle(image, (x, y0), (x + tw + 6, y0 + th + baseline + 4), color, -1)
    cv2.putText(image, text, (x + 3, y0 + th + 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_annotations(image, annotations, category_names, line_width=2):
    color = (0, 255, 0)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)

        category_id = ann.get("category_id")
        label = category_names.get(category_id, f"id={category_id}")
        if "score" in ann:
            label = f"{label} {float(ann['score']):.2f}"
        draw_label(image, label, x1, y1, color)


def visualize(args):
    coco = load_coco(args.coco_ann)
    fred_root = Path(args.fred_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_names = {
        item["id"]: item.get("name", str(item["id"]))
        for item in coco.get("categories", [])
    }

    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    selected_images = [
        image_info
        for image_info in coco.get("images", [])
        if image_matches(image_info, split=args.split, seq_name=args.seq_name)
    ]

    if args.only_with_annotations:
        selected_images = [
            image_info
            for image_info in selected_images
            if anns_by_image.get(image_info["id"])
        ]

    selected_images = sorted(selected_images, key=lambda item: normalize_rel_path(item.get("file_name", "")))
    if args.max_images is not None:
        selected_images = selected_images[: args.max_images]

    print(f"Loaded images: {len(coco.get('images', []))}")
    print(f"Loaded annotations: {len(coco.get('annotations', []))}")
    print(f"Selected images: {len(selected_images)}")
    print(f"Output dir: {output_dir}")

    saved = 0
    missing = 0
    empty = 0

    for image_info in selected_images:
        rel_path = normalize_rel_path(image_info["file_name"])
        image_path = fred_root / rel_path
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: could not read image: {image_path}")
            missing += 1
            continue

        annotations = anns_by_image.get(image_info["id"], [])
        if not annotations:
            empty += 1

        draw_annotations(image, annotations, category_names, line_width=args.line_width)

        safe_name = rel_path.replace("/", "__")
        output_path = output_dir / safe_name
        cv2.imwrite(str(output_path), image)
        saved += 1

    print(f"Saved visualizations: {saved}")
    print(f"Images without annotations: {empty}")
    print(f"Missing/unreadable images: {missing}")


def main():
    parser = argparse.ArgumentParser(description="Visualize COCO annotations on FRED event frames.")
    parser.add_argument("--coco-ann", required=True, help="Path to train.json/test.json or jsonl annotation file.")
    parser.add_argument("--fred-root", required=True, help="Root directory of FRED dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualized images.")
    parser.add_argument("--split", default=None, help="Optional split filter, for example train or test.")
    parser.add_argument("--seq-name", default=None, help="Optional FRED sequence id/name, for example 0.")
    parser.add_argument("--max-images", type=int, default=200, help="Maximum number of images to visualize.")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width.")
    parser.add_argument("--only-with-annotations", action="store_true", help="Skip images without annotations.")
    args = parser.parse_args()

    visualize(args)


if __name__ == "__main__":
    main()
