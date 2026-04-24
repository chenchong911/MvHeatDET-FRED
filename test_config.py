"""
测试完整的FRED配置
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from src.core import YAMLConfig


def test_complete_config():
    print("Testing complete FRED configuration...")
    
    # 加载主配置文件
    cfg = YAMLConfig('configs/fred_complete.yml')
    print("✓ Main configuration loaded")
    
    # 尝试访问train_dataloader属性，这会触发数据加载器的创建
    try:
        train_loader = cfg.train_dataloader
        print(f"✓ Train dataloader created successfully")
        return True
    except Exception as e:
        print(f"✗ Error creating train dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_config()
    if success:
        print("\n✓ Configuration test passed!")
    else:
        print("\n✗ Configuration test failed!")