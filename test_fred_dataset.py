"""
测试FRED数据集加载
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from src.core import YAMLConfig
import torch
from torch.utils.data import DataLoader

def test_fred_dataset():
    print("Testing FRED dataset configuration...")
    
    # 加载配置 - 使用具体的FRED数据集配置文件
    cfg = YAMLConfig('configs/dataset/FRED_detection.yml')
    print("✓ Configuration loaded successfully")
    
    # 检查配置结构
    print(f"Available config keys: {list(cfg.yaml_cfg.keys())}")
    
    # 检查数据集配置
    train_dataloader = cfg.yaml_cfg.get('train_dataloader', {})
    dataset_config = train_dataloader.get('dataset', {})
    print(f"Dataset config: {dataset_config}")
    
    if 'type' in dataset_config:
        print(f"✓ Dataset type: {dataset_config['type']}")
        print(f"✓ Dataset path: {dataset_config.get('img_folder', 'N/A')}")
        
        # 检查数据集是否存在
        dataset_path = dataset_config.get('img_folder', '')
        if os.path.exists(dataset_path):
            print(f"✓ Dataset path exists: {dataset_path}")
        else:
            print(f"✗ Dataset path does not exist: {dataset_path}")
            return False
        
        # 尝试创建数据集实例
        try:
            # 获取数据集类
            from src.core.yaml_utils import create
            dataset_obj = create(dataset_config)
            print(f"✓ Dataset object created successfully with {len(dataset_obj)} samples")
            
            # 尝试获取一个样本
            if len(dataset_obj) > 0:
                sample_img, sample_target = dataset_obj[0]
                print(f"✓ Sample loaded - Image shape: {sample_img.shape}")
                print(f"✓ Sample loaded - Target keys: {list(sample_target.keys())}")
            
            return True
        except Exception as e:
            print(f"✗ Error creating dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("✗ Dataset config doesn't have 'type' key")
        print("Now trying with the full configuration")
        
        # 使用完整的配置文件
        full_cfg = YAMLConfig('configs/fred_complete.yml')
        full_train_dataloader = full_cfg.yaml_cfg.get('train_dataloader', {})
        full_dataset_config = full_train_dataloader.get('dataset', {})
        
        print(f"Full dataset config: {full_dataset_config}")
        
        if 'type' in full_dataset_config:
            print(f"✓ Dataset type: {full_dataset_config['type']}")
            print(f"✓ Dataset path: {full_dataset_config.get('img_folder', 'N/A')}")
            print(f"✓ Annotation file: {full_dataset_config.get('ann_file', 'N/A')}")
            
            # 检查数据集和标注文件是否存在
            dataset_path = full_dataset_config.get('img_folder', '')
            ann_file = full_dataset_config.get('ann_file', '')
            
            if os.path.exists(dataset_path):
                print(f"✓ Dataset path exists: {dataset_path}")
            else:
                print(f"✗ Dataset path does not exist: {dataset_path}")
                return False
                
            if os.path.exists(ann_file):
                print(f"✓ Annotation file exists: {ann_file}")
            else:
                print(f"✗ Annotation file does not exist: {ann_file}")
                return False
            
            # 尝试创建数据集实例
            try:
                from src.core.yaml_utils import create
                dataset_obj = create(full_dataset_config)
                print(f"✓ Dataset object created successfully with {len(dataset_obj)} samples")
                
                # 尝试获取一个样本
                if len(dataset_obj) > 0:
                    sample_img, sample_target = dataset_obj[0]
                    print(f"✓ Sample loaded - Image shape: {sample_img.shape}")
                    print(f"✓ Sample loaded - Target keys: {list(sample_target.keys())}")
                
                return True
            except Exception as e:
                print(f"✗ Error creating dataset: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("✗ Full dataset config also doesn't have 'type' key")
            return False

if __name__ == "__main__":
    success = test_fred_dataset()
    if success:
        print("\n✓ All tests passed! You can now run training with: python tools/train.py -c configs/fred_complete.yml")
    else:
        print("\n✗ Tests failed!")