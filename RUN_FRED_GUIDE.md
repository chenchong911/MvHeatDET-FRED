# Training MvHeatDET on FRED Dataset

This guide explains how to train the MvHeatDET model on the FRED dataset.

## Prerequisites

- Ensure the FRED dataset is available at `/mnt/data/cc/FRED/`
- Make sure you have the required dependencies installed (`pip install -r requirements.txt`)
- The FRED dataset should contain train and test splits with event frames in `Event/Frames` subdirectories

## Steps to Train

### 1. Convert FRED Dataset to COCO Format

First, convert the FRED dataset to COCO format which is expected by MvHeatDET:

```bash
# From the MvHeatDET-FRED directory
cd tools
python convert_all_fred_to_coco.py
```

Or using the shell script:
```bash
bash run_fred_training.sh prepare_data
```

This will create annotation files in `/mnt/data/cc/FRED/coco_annotations/`.

### 2. Train the Model

Once the data is prepared, you can start training:

```bash
# Using the Python script
python run_fred_training.py --train --data-path /mnt/data/cc/FRED

# Or using the shell script
bash run_fred_training.sh train
```

### 3. Full Pipeline (Prepare + Train)

To run the complete pipeline in one go:

```bash
# Using the Python script
python run_fred_training.py --prepare-data --train --data-path /mnt/data/cc/FRED

# Or using the shell script
bash run_fred_training.sh full
```

### 4. Test the Model

To evaluate a trained model:

```bash
# Using the Python script
python run_fred_training.py --test --checkpoint /path/to/checkpoint --data-path /mnt/data/cc/FRED

# Or using the shell script
bash run_fred_training.sh test --checkpoint /path/to/checkpoint
```

## Configuration Details

- The model expects event frames as input (from the `Event/Frames` directory)
- The model is configured for 1 class ("drone") instead of the original 10 classes
- Output will be saved to `/mnt/data/cc/FRED_output/MvHeatDET`
- Input size is set to 640x640 pixels

## Notes

- The conversion script parses `coordinates.txt` files to extract bounding box annotations
- The model uses the same architecture as the original MvHeatDET but adapted for a single class
- Training logs and checkpoints will be saved to the output directory specified in the config
- For multi-GPU training, use the torchrun command with the appropriate parameters