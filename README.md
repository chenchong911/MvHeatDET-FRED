# MvHeatDET-FRED

[English](./README.md) | [简体中文](./README_zh-CN.md)

`MvHeatDET-FRED` is an adaptation of `MvHeatDET` for the `FRED` event-camera drone detection dataset. The repository keeps the original MvHeatDET detection framework, replaces the dataset side with a FRED-specific COCO-style pipeline, and provides conversion scripts plus training/testing entrypoints for the FRED setting.

This repo is suitable if you want to:

- convert raw FRED sequences into COCO annotations,
- train the MvHeatDET detector on a single-class drone dataset,
- evaluate checkpoints with the same detection pipeline,
- understand how the FRED adaptation is wired into the original codebase.

## What Is In This Repository

The core training stack is:

- `MvHeat_DET` backbone in [src/zoo/evheat/vHeat_MoE.py](/e:/ChenChong/project/MvHeatDET-FRED/src/zoo/evheat/vHeat_MoE.py)
- `RT-DETR` style decoder, matcher, criterion, and postprocessor in [src/zoo/rtdetr](/e:/ChenChong/project/MvHeatDET-FRED/src/zoo/rtdetr)
- YAML-driven config and object registry in [src/core](/e:/ChenChong/project/MvHeatDET-FRED/src/core)
- FRED dataset loader in [src/data/fred_dataset.py](/e:/ChenChong/project/MvHeatDET-FRED/src/data/fred_dataset.py)
- unified train/test entry in [tools/train.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/train.py)

Compared with the original EvDET200K-oriented configuration, this FRED version mainly changes:

- dataset definition to `FREDDetection`,
- number of classes to `1`,
- data paths and COCO annotation files,
- FRED conversion scripts under [tools](/e:/ChenChong/project/MvHeatDET-FRED/tools).

## Project Structure

```text
MvHeatDET-FRED/
|-- configs/
|   |-- fred_complete.yml               # Main FRED training config
|   |-- dataset/FRED_detection.yml      # Dataset-only config
|   `-- evheat/include/                 # Model, optimizer, dataloader includes
|-- src/
|   |-- core/                           # YAML config and registry system
|   |-- data/                           # Datasets, transforms, dataloader
|   |-- solver/                         # Training and evaluation loop
|   |-- zoo/                            # Backbone, decoder, matcher, criterion
|   `-- misc/                           # Distributed utils, logging, visualization
|-- tools/
|   |-- train.py                        # Main training/testing entry
|   |-- convert_fred_to_coco.py         # Single split converter
|   |-- convert_all_fred_to_coco.py     # Train/test conversion wrapper
|   `-- export_onnx.py                  # ONNX export helper
|-- run_fred_training.py                # Convenience pipeline runner
|-- run_fred_training.sh                # Shell wrapper
|-- test_fred_dataset.py                # Dataset sanity check
`-- README_zh-CN.md                     # Previous Chinese documentation
```

## Code Reading Guide

If you want to understand the repo quickly, this is the shortest useful reading path:

1. Start from [tools/train.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/train.py).
   It parses `--config`, `--resume`, `--test-only`, and `--amp`, then builds `YAMLConfig` and dispatches to the task solver.
2. Read [src/core/yaml_config.py](/e:/ChenChong/project/MvHeatDET-FRED/src/core/yaml_config.py).
   This is the center of the config-driven design: it loads YAML, merges includes, and instantiates model, dataloaders, optimizer, EMA, and AMP objects on demand.
3. Read [src/solver/solver.py](/e:/ChenChong/project/MvHeatDET-FRED/src/solver/solver.py) and [src/solver/det_solver.py](/e:/ChenChong/project/MvHeatDET-FRED/src/solver/det_solver.py).
   These files define setup, checkpoint loading, training, validation, and best-checkpoint saving.
4. Read [src/data/fred_dataset.py](/e:/ChenChong/project/MvHeatDET-FRED/src/data/fred_dataset.py).
   This is the FRED-specific dataset wrapper over the generic COCO loader.
5. Read [configs/fred_complete.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/fred_complete.yml) and its included files.
   This tells you the actual training recipe used by the FRED adaptation.
6. Read [tools/convert_fred_to_coco.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/convert_fred_to_coco.py).
   This shows the expected FRED raw directory layout and how `coordinates.txt` becomes COCO annotations.

## How The FRED Version Works

### 1. Dataset format

The repository expects raw FRED data to look like:

```text
FRED/
|-- train/
|   `-- <sequence_name>/
|       |-- coordinates.txt
|       `-- Event/Frames/*.png
`-- test/
    `-- <sequence_name>/
        |-- coordinates.txt
        `-- Event/Frames/*.png
```

The conversion script reads:

- frame images from `Event/Frames/*.png`
- bounding boxes from `coordinates.txt`

and writes:

- `coco_annotations/train.json`
- `coco_annotations/test.json`

### 2. Annotation conversion

[tools/convert_fred_to_coco.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/convert_fred_to_coco.py) converts one split at a time. It:

- scans all sequences under `train` or `test`,
- parses each line of `coordinates.txt`,
- creates one COCO image item per event frame,
- creates one COCO bbox annotation per matching timestamp,
- uses a single category: `drone` with `category_id = 1`.

[tools/convert_all_fred_to_coco.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/convert_all_fred_to_coco.py) is a wrapper that converts both splits.

### 3. Dataset loader

[src/data/fred_dataset.py](/e:/ChenChong/project/MvHeatDET-FRED/src/data/fred_dataset.py) subclasses the generic COCO dataset loader and mainly does two things:

- resolves FRED image paths correctly from COCO `file_name`,
- keeps empty-box samples safe by forcing empty bounding boxes to shape `[0, 4]`.

### 4. Training pipeline

The actual training entry is:

```bash
python tools/train.py -c configs/fred_complete.yml
```

Internally the flow is:

`tools/train.py` -> `YAMLConfig` -> `DetSolver` -> `train_one_epoch` / `evaluate`

### 5. Model configuration

The FRED-specific model recipe lives in [configs/evheat/include/mvheatdet_fred.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/mvheatdet_fred.yml):

- backbone: `MvHeat_DET`
- decoder: `RTDETRTransformer`
- criterion: `SetCriterion`
- postprocessor: `RTDETRPostProcessor`
- `num_classes: 1`
- input size: `640 x 640`
- queries: `100`
- decoder layers: `6`

## Installation

Recommended environment from the repository:

```bash
conda create -n mvheatfred python=3.8
conda activate mvheatfred
pip install -r requirements.txt
```

Main dependencies in [requirements.txt](/e:/ChenChong/project/MvHeatDET-FRED/requirements.txt):

- `torch==2.0.1`
- `torchvision==0.15.2`
- `pycocotools`
- `timm`
- `einops`
- `transformers`
- `thop`

## Data Preparation

### Option 1: Convert train and test together

```bash
cd tools
python convert_all_fred_to_coco.py
```

### Option 2: Convert one split manually

```bash
python tools/convert_fred_to_coco.py \
  --fred-root /path/to/FRED \
  --output-dir /path/to/FRED/coco_annotations \
  --split train

python tools/convert_fred_to_coco.py \
  --fred-root /path/to/FRED \
  --output-dir /path/to/FRED/coco_annotations \
  --split test
```

After conversion, verify that:

- `/path/to/FRED/coco_annotations/train.json` exists
- `/path/to/FRED/coco_annotations/test.json` exists

You can also run:

```bash
python test_fred_dataset.py
```

## Training

### Standard training

```bash
python tools/train.py -c configs/fred_complete.yml
```

### Resume training

```bash
python tools/train.py -c configs/fred_complete.yml -r /path/to/checkpoint.pth
```

### Enable AMP

```bash
python tools/train.py -c configs/fred_complete.yml --amp
```

### Multi-GPU training

```bash
torchrun --nproc_per_node=2 tools/train.py -c configs/fred_complete.yml --amp
```

The current FRED config uses:

- batch size: `6`
- epochs: `80`
- optimizer: `AdamW`
- learning rate: `1e-4`
- EMA: enabled

## Evaluation

```bash
python tools/train.py -c configs/fred_complete.yml -r /path/to/checkpoint.pth --test-only
```

Evaluation uses the validation dataloader defined in [configs/evheat/include/train_dataloader.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/train_dataloader.yml), which points to the FRED `test` split.

## Convenience Scripts

Two helper scripts are included:

- [run_fred_training.py](/e:/ChenChong/project/MvHeatDET-FRED/run_fred_training.py)
- [run_fred_training.sh](/e:/ChenChong/project/MvHeatDET-FRED/run_fred_training.sh)

Example usage:

```bash
python run_fred_training.py --prepare-data --train --data-path /mnt/data/cc/FRED
python run_fred_training.py --test --checkpoint /path/to/checkpoint.pth --data-path /mnt/data/cc/FRED
```

## Important Path Notes

Several files currently assume Linux-style absolute paths by default:

- [configs/fred_complete.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/fred_complete.yml)
- [configs/dataset/FRED_detection.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/dataset/FRED_detection.yml)
- [configs/evheat/include/train_dataloader.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/train_dataloader.yml)
- [run_fred_training.py](/e:/ChenChong/project/MvHeatDET-FRED/run_fred_training.py)
- [tools/convert_all_fred_to_coco.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/convert_all_fred_to_coco.py)

By default they point to:

```text
/mnt/data/cc/FRED
/mnt/data/cc/FRED/coco_annotations
/mnt/data/cc/FRED_output/MvHeatDET
```

If your dataset is stored elsewhere, update those paths before training.

## Key Config Files

- [configs/fred_complete.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/fred_complete.yml)
  Main FRED entry config. Includes dataset, runtime, dataloader, optimizer, and model settings.
- [configs/dataset/FRED_detection.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/dataset/FRED_detection.yml)
  Dataset-only config with `FREDDetection`.
- [configs/evheat/include/train_dataloader.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/train_dataloader.yml)
  Training and validation transforms plus dataloader options.
- [configs/evheat/include/optimizer.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/optimizer.yml)
  Epoch count, optimizer, scheduler, EMA, gradient clipping.
- [configs/evheat/include/mvheatdet_fred.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/mvheatdet_fred.yml)
  Model architecture and detection loss/postprocess settings.

## Common Issues

### No annotation files found

Make sure you have already generated:

- `coco_annotations/train.json`
- `coco_annotations/test.json`

### Dataset path exists but images do not load

Check that the raw FRED directory still contains:

- `train/<sequence>/Event/Frames/*.png`
- `test/<sequence>/Event/Frames/*.png`

The COCO json stores relative `file_name` values like:

```text
train/<sequence>/Event/Frames/<frame>.png
```

### Training starts but crashes on file paths

This is usually caused by hard-coded paths in config files or helper scripts. Search and replace `/mnt/data/cc/FRED` with your local dataset root.

### Empty-box or shape-related batch errors

The repository already includes FRED-specific handling for empty boxes in:

- [src/data/fred_dataset.py](/e:/ChenChong/project/MvHeatDET-FRED/src/data/fred_dataset.py)
- [src/solver/det_engine.py](/e:/ChenChong/project/MvHeatDET-FRED/src/solver/det_engine.py)

If you modify the dataset or transforms, keep that behavior intact.

## Acknowledgement

This repository is based on the original MvHeatDET / RT-DETR style codebase and extends it for FRED dataset training and evaluation.

## Citation

If you use the original MvHeatDET method, please cite the related paper:

```bibtex
@misc{wang2024EvDET200K,
  title={Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset},
  author={Xiao Wang and Yu Jin and Wentao Wu and Wei Zhang and Lin Zhu and Bo Jiang and Yonghong Tian},
  year={2024},
  eprint={2412.06647},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2412.06647}
}
```
