"""
Custom FRED dataset loader adapted for MvHeatDET
"""

import torch
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from natsort import natsorted
import tqdm
import json
import copy

from src.core import register
from src.data.coco.coco_dataset import ConvertCocoPolysToMask, CocoDetection

__all__ = ['FREDDetection']

@register
class FREDDetection(CocoDetection):
    """
    适用于MvHeatDET的FRED数据集加载器，基于COCO格式
    """
    __inject__ = ['transforms']
    
    def __init__(self, img_folder, ann_file, transforms=None, return_masks=False, remap_mscoco_category=False):
        # 使用父类CocoDetection的初始化
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category
        )
        
        # Set split based on path
        if 'train' in img_folder:
            self.split = 'train'
        elif 'test' in img_folder:
            self.split = 'test'
        else:
            self.split = 'unknown'
            
        print(f"FRED {self.split} dataset initialized with {len(self.ids)} samples.")
        if len(self.ids) == 0:
            print(f"WARNING: No valid samples found in {img_folder} with annotations {ann_file}!")
            print(f"Check if the paths exist and contain valid COCO format annotations.")
    
    def _load_image(self, id):
        """
        Override the _load_image method to handle FRED dataset path structure
        """
        path = self.coco.loadImgs(id)[0]["file_name"]
        # Remove the leading 'train/' or 'test/' from the path
        if path.startswith('train/'):
            path = path[6:]
        elif path.startswith('test/'):
            path = path[5:]
        full_path = os.path.join(self.root, path)
        return Image.open(full_path).convert("RGB")
    
    def __getitem__(self, idx):
        """
        Override getitem to ensure consistent bbox shape
        """
        img, target = super().__getitem__(idx)
        
        # 确保bbox张量的形状一致
        if 'boxes' in target and target['boxes'].numel() == 0:
            # 如果bbox为空，创建一个形状为[0, 4]的空张量
            target['boxes'] = torch.zeros((0, 4), dtype=target['boxes'].dtype, device=target['boxes'].device)
        
        return img, target