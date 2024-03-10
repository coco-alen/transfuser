# CUDA_VISIBLE_DEVICES=0 python train_new.py \
#     --id vitfuser_noEmbed \
#     --epochs 60 \
#     --lr 0.0001 \
#     --val_every 1 \
#     --batch_size 256 \
#     --gpus 1 \
#     --load_weights "/home/gyp/program/my_transfuser/transfuser/TCP/log/vitfuser/best_epoch=52-val_loss=0.784.ckpt"


CUDA_VISIBLE_DEVICES=0 python train.py \
    --id tcp_int4 \
    --epochs 60 \
    --lr 0.0001 \
    --val_every 1 \
    --batch_size 32 \
    --gpus 1 \
    --load_weights "/home/gyp/program/TCP/log/TCP/best_epoch=16-val_loss=0.747.ckpt" \
    --w_bits 4 \
    --a_bits 8 \
    --symmetric
