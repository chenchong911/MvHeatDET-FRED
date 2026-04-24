import os
from PIL import Image
import glob

def check_resolution():
    # 选取train中的几张图片检查分辨率，使用更具体的路径模式
    train_img_paths = []
    # 先找到几个序列目录
    sequence_dirs = [d for d in os.listdir('/mnt/data/cc/FRED/train/') if os.path.isdir(os.path.join('/mnt/data/cc/FRED/train/', d))]
    
    for seq_dir in sequence_dirs[:3]:  # 检查前3个序列
        frames_dir = f'/mnt/data/cc/FRED/train/{seq_dir}/Event/Frames/'
        if os.path.exists(frames_dir):
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
            for frame_file in frame_files[:2]:  # 每个序列检查前2张图片
                train_img_paths.append(os.path.join(frames_dir, frame_file))
    
    print("Checking image resolutions in FRED dataset:")
    resolutions = set()
    
    for path in train_img_paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
                resolutions.add((width, height))
                print(f'{os.path.basename(path)}: {width}x{height}')
        except Exception as e:
            print(f"Error reading {path}: {e}")
    
    print(f"\nUnique resolutions found: {resolutions}")
    
    # 检查test集
    test_img_paths = []
    test_sequence_dirs = [d for d in os.listdir('/mnt/data/cc/FRED/test/') if os.path.isdir(os.path.join('/mnt/data/cc/FRED/test/', d))]
    
    for seq_dir in test_sequence_dirs[:2]:  # 检查前2个序列
        frames_dir = f'/mnt/data/cc/FRED/test/{seq_dir}/Event/Frames/'
        if os.path.exists(frames_dir):
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
            for frame_file in frame_files[:2]:  # 每个序列检查前2张图片
                test_img_paths.append(os.path.join(frames_dir, frame_file))
    
    print("\nChecking test set images:")
    for path in test_img_paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
                resolutions.add((width, height))
                print(f'{os.path.basename(path)}: {width}x{height}')
        except Exception as e:
            print(f"Error reading {path}: {e}")
    
    print(f"\nAll unique resolutions in dataset: {resolutions}")

if __name__ == "__main__":
    check_resolution()