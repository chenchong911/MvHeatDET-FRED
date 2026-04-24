"""
检查COCO JSON文件
"""
import json
import os

def check_coco_json():
    train_json_path = '/mnt/data/cc/FRED/coco_annotations/train.json'
    test_json_path = '/mnt/data/cc/FRED/coco_annotations/test.json'
    
    print("Checking COCO annotation files...")
    
    if os.path.exists(train_json_path):
        print(f"✓ Train JSON file exists: {train_json_path}")
        with open(train_json_path, 'r') as f:
            try:
                train_data = json.load(f)
                print(f"✓ Train JSON is valid, keys: {list(train_data.keys())}")
                if 'images' in train_data:
                    print(f"✓ Train JSON has {len(train_data['images'])} images")
                if 'annotations' in train_data:
                    print(f"✓ Train JSON has {len(train_data['annotations'])} annotations")
            except Exception as e:
                print(f"✗ Error loading train JSON: {e}")
    else:
        print(f"✗ Train JSON file does not exist: {train_json_path}")
    
    if os.path.exists(test_json_path):
        print(f"✓ Test JSON file exists: {test_json_path}")
        with open(test_json_path, 'r') as f:
            try:
                test_data = json.load(f)
                print(f"✓ Test JSON is valid, keys: {list(test_data.keys())}")
                if 'images' in test_data:
                    print(f"✓ Test JSON has {len(test_data['images'])} images")
                if 'annotations' in test_data:
                    print(f"✓ Test JSON has {len(test_data['annotations'])} annotations")
            except Exception as e:
                print(f"✗ Error loading test JSON: {e}")
    else:
        print(f"✗ Test JSON file does not exist: {test_json_path}")

if __name__ == "__main__":
    check_coco_json()