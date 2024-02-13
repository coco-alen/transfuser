CUDA_VISIBLE_DEVICES=0 python train.py \
    --id transfuser_linearAttn \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.0001 \
    --load_weight /home/yipin/program/transfuser/model_ckpt/2021/transfuser/best_model.pth