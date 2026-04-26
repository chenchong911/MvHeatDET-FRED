"""by lyuwenyu
"""

import sys 
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision

from collections import OrderedDict
from typing import List, Tuple, Union, Any

import src.misc.dist as dist
from src.misc import MetricLogger, SmoothedValue, reduce_dict
from src.data.coco.coco_eval import CocoEvaluator
from src.data.coco.coco_utils import get_coco_api_from_dataset


def train_one_epoch(model: nn.Module, criterion: nn.Module, 
        data_loader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, 
        clip_max_norm: float = 0, print_freq: int = 100, ema=None, scaler=None):
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        # 确保targets是字典列表
        if isinstance(targets, list) and len(targets) > 0:
            if isinstance(targets[0], dict):
                # 处理每个target中的张量
                processed_targets = []
                for t in targets:
                    processed_t = {}
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            # 对于bbox等可能为空的张量，确保它们有正确的形状
                            if k == 'boxes' and v.numel() == 0:
                                # 创建一个空的bbox张量，形状为[0, 4]
                                processed_t[k] = torch.zeros((0, 4), dtype=v.dtype, device=device)
                            else:
                                processed_t[k] = v.to(device)
                        else:
                            processed_t[k] = v
                    processed_targets.append(processed_t)
                targets = processed_targets
            elif isinstance(targets[0], str):
                # 如果targets是字符串，可能是错误的数据格式
                raise ValueError(f"Invalid targets format: {type(targets[0])}")
        
        if scaler is not None:
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(samples, targets)
                losses = criterion(outputs, targets)
            
            loss = losses['loss']
            scaler.scale(loss).backward()
            
            if clip_max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        else:
            outputs = model(samples, targets)
            losses = criterion(outputs, targets)

            loss = losses['loss']
            optimizer.zero_grad()
            loss.backward()
            
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            
            optimizer.step()
        
        if ema:
            ema.update(model)
        
        loss_dict = losses.copy()
        loss_dict.pop('loss')
        loss_dict = reduce_dict(loss_dict)
        loss_dict['loss'] = losses['loss'].clone()
        
        metric_logger.update(**loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # 确保targets是字典列表
        if isinstance(targets, list) and len(targets) > 0:
            if isinstance(targets[0], dict):
                # 处理每个target中的张量
                processed_targets = []
                for t in targets:
                    processed_t = {}
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            # 对于bbox等可能为空的张量，确保它们有正确的形状
                            if k == 'boxes' and v.numel() == 0:
                                # 创建一个空的bbox张量，形状为[0, 4]
                                processed_t[k] = torch.zeros((0, 4), dtype=v.dtype, device=device)
                            else:
                                processed_t[k] = v.to(device)
                        else:
                            processed_t[k] = v
                    processed_targets.append(processed_t)
                targets = processed_targets
            elif isinstance(targets[0], str):
                # 如果targets是字符串，可能是错误的数据格式
                raise ValueError(f"Invalid targets format: {type(targets[0])}")

        outputs = model(samples, targets)
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessors(outputs, orig_target_sizes)
        
        # Evaluate
        if dist.is_main_process():
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            evaluator_type = getattr(coco_evaluator.coco_eval['bbox'].params, 'evaluator_type', None)
            if evaluator_type is not None and evaluator_type == 'deteval':
                coco_evaluator.update_deteval(res)
            else:
                coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # accumulate predictions from all images
    coco_evaluator.synchronize_between_processes()
    
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        stats[f'coco_eval_{iou_type}'] = coco_eval.stats
    
    return stats, coco_evaluator
