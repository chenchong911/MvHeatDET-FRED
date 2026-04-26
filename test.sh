# train on multi-gpu
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
# torchrun --nproc_per_node=2 tools/train.py -c configs/fred_complete.yml \
#                         -r /mnt/data/cc/FRED_output/MvHeatDET/checkpoint.pth \
#                         --test-only


# test on single-gpu 
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py -c configs/fred_complete.yml \
                        -r /mnt/data/cc/FRED_output/MvHeatDET/checkpoint.pth \
                        --test-only
