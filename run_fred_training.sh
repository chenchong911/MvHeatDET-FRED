#!/bin/bash
set -euo pipefail

# Script to run MvHeatDET training on FRED dataset

echo "FRED dataset training script for MvHeatDET"
echo "========================================="

# Check if the script is called with an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 {prepare_data|train|test|full} [options]"
    echo "  prepare_data - Convert FRED dataset to COCO format"
    echo "  train        - Start training on FRED dataset"
    echo "  test         - Start testing on FRED dataset (requires checkpoint path)"
    echo "  full         - Full pipeline: prepare_data -> train"
    echo ""
    echo "Options:"
    echo "  --checkpoint PATH - Path to checkpoint file (required for test)"
    echo "  --resume PATH     - Path to checkpoint file (optional for train/full)"
    echo "  --amp             - Enable Automatic Mixed Precision training"
    echo ""
    exit 1
fi

COMMAND=$1
shift
CHECKPOINT_PATH=""
RESUME_PATH=""
USE_AMP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        --amp)
            USE_AMP="--amp"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

case $COMMAND in
    prepare_data)
        echo "Preparing FRED dataset..."
        python tools/convert_all_fred_to_coco.py
        ;;
    train)
        echo "Starting training on FRED dataset..."
        cd tools
        if [ -n "$RESUME_PATH" ]; then
            python train.py -c ../configs/fred_complete.yml -r "$RESUME_PATH" $USE_AMP
        else
            python train.py -c ../configs/fred_complete.yml $USE_AMP
        fi
        ;;
    test)
        if [ -z "$CHECKPOINT_PATH" ]; then
            echo "Error: --checkpoint PATH is required for testing"
            exit 1
        fi
        echo "Starting testing on FRED dataset with checkpoint: $CHECKPOINT_PATH"
        cd tools && python train.py -c ../configs/fred_complete.yml -r "$CHECKPOINT_PATH" --test-only $USE_AMP
        ;;
    full)
        echo "Running full pipeline: prepare_data -> train"
        echo "Step 1: Preparing FRED dataset..."
        python tools/convert_all_fred_to_coco.py
        
        echo "Step 2: Starting training on FRED dataset..."
        cd tools
        if [ -n "$RESUME_PATH" ]; then
            python train.py -c ../configs/fred_complete.yml -r "$RESUME_PATH" $USE_AMP
        else
            python train.py -c ../configs/fred_complete.yml $USE_AMP
        fi
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Available commands: prepare_data, train, test, full"
        exit 1
        ;;
esac

echo "Completed successfully."