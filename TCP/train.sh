CUDA_VISIBLE_DEVICES=0 python train_new.py \
    --id vitfuser \
    --epochs 60 \
    --lr 0.0005 \
    --val_every 1 \
    --batch_size 256 \
    --gpus 1