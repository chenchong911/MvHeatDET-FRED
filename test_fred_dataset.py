"""Sanity-check the FRED dataloader defined by configs/fred_complete.yml."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from src.core import YAMLConfig
from src.core.yaml_utils import create, merge_config


def unwrap_sequence_subset(dataset_config):
    """Return the inner FREDDetection config when SequenceDatasetSubset is used."""
    if dataset_config.get("type") == "SequenceDatasetSubset":
        return dataset_config.get("dataset", {})
    return dataset_config


def test_fred_dataset():
    print("Testing FRED dataset configuration...")

    cfg = YAMLConfig("configs/fred_complete.yml")
    merge_config(cfg.yaml_cfg)
    print("OK: configuration loaded from configs/fred_complete.yml")

    loader_config = cfg.yaml_cfg.get("train_dataloader", {})
    dataset_config = loader_config.get("dataset", {})
    inner_dataset_config = unwrap_sequence_subset(dataset_config)

    print(f"Dataset wrapper: {dataset_config.get('type', 'N/A')}")
    print(f"Dataset type: {inner_dataset_config.get('type', 'N/A')}")
    print(f"Dataset path: {inner_dataset_config.get('img_folder', 'N/A')}")
    print(f"Annotation file: {inner_dataset_config.get('ann_file', 'N/A')}")

    dataset_path = inner_dataset_config.get("img_folder", "")
    ann_file = inner_dataset_config.get("ann_file", "")

    if not os.path.exists(dataset_path):
        print(f"ERROR: dataset path does not exist: {dataset_path}")
        return False

    if not os.path.exists(ann_file):
        print(f"ERROR: annotation file does not exist: {ann_file}")
        return False

    try:
        dataset_obj = create(dataset_config)
        print(f"OK: dataset object created with {len(dataset_obj)} samples")

        if len(dataset_obj) > 0:
            sample_img, sample_target = dataset_obj[0]
            print(f"OK: first image shape: {sample_img.shape}")
            print(f"OK: first target keys: {list(sample_target.keys())}")

        return True
    except Exception as exc:
        print(f"ERROR: failed to create or read dataset: {exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_fred_dataset()
    if success:
        print("\nAll checks passed. You can train with: python tools/train.py -c configs/fred_complete.yml")
    else:
        print("\nChecks failed.")
