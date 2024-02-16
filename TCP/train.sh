CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --id tcp_256_2 \
    --epochs 60 \
    --lr 0.0001 \
    --val_every 1 \
    --batch_size 64 \
    --gpus 2 \
    --load_weights "/home/gyp/program/TCP/log/tcp_256/epoch=32-last.ckpt"