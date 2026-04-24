'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data.coco.coco_utils import get_coco_api_from_dataset
from src.core import register

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from thop import profile

@register
class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # base_ds = self.val_dataloader.dataset
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, 'coco_eval_bbox': float('-inf')}

        start_time = time.time()

        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # Keep periodic checkpoints every 10 epochs.
                if (epoch + 1) % 10 == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, _ = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            if 'coco_eval_bbox' in test_stats:
                bbox_ap = float(test_stats['coco_eval_bbox'][0])
                if bbox_ap > best_stat['coco_eval_bbox']:
                    best_stat['epoch'] = epoch
                    best_stat['coco_eval_bbox'] = bbox_ap
                    if self.output_dir:
                        dist.save_on_master(self.state_dict(epoch), self.output_dir / 'best.pth')
            
            print(f'Best COCO eval bbox: {best_stat["coco_eval_bbox"]:.4f} at epoch: {best_stat["epoch"]}')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        # Close writer
        if self.cfg.use_tensorboard:
            self.cfg.writer.close()

    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return