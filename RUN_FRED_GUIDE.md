# Training MvHeatDET on FRED Dataset

This guide keeps the FRED workflow explicit: convert annotations with `tools/convert_all_fred_to_coco.py`, then train or evaluate with `tools/train.py`.

## Prerequisites

- FRED is available at `/mnt/data/cc/FRED/`.
- Dependencies are installed with `pip install -r requirements.txt`.
- Each FRED sequence contains event frames under `Event/Frames` and annotations in `coordinates.txt`.

## 1. Convert FRED to COCO

```bash
python tools/convert_all_fred_to_coco.py \
  --fred-root /mnt/data/cc/FRED \
  --output-dir /mnt/data/cc/FRED/coco_annotations \
  --image-width 1280 \
  --image-height 720
```

This generates:

- `/mnt/data/cc/FRED/coco_annotations/train.json`
- `/mnt/data/cc/FRED/coco_annotations/test.json`

## 2. Train

```bash
python tools/train.py -c configs/fred_complete.yml
```

Resume from a checkpoint:

```bash
python tools/train.py -c configs/fred_complete.yml -r /path/to/checkpoint.pth
```

Enable AMP:

```bash
python tools/train.py -c configs/fred_complete.yml --amp
```

## 3. Evaluate

```bash
python tools/train.py \
  -c configs/fred_complete.yml \
  -r /path/to/checkpoint.pth \
  --test-only
```

## Notes

- Data paths and sequence sampling live in `configs/evheat/include/train_dataloader.yml`.
- Model size and decoder settings live in `configs/evheat/include/mvheatdet_fred.yml`.
- Training logs and checkpoints are saved to the `output_dir` in `configs/fred_complete.yml`.
