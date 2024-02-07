CUDA_VISIBLE_DEVICES=0 python train.py \
    --id transfuser_focusview \
    --batch_size 64 \
    --epochs 10 \
    --lr 0.00001 \
    --load_weight /home/yipin/program/transfuser/model_ckpt/2021/transfuser/best_model.pth