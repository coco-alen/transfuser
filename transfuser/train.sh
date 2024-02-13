CUDA_VISIBLE_DEVICES=0 python train.py \
    --id transfuser_focusview \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.0001
    # --load_weight /home/gyp/program/my_transfuser/transfuser/model_ckpt/2021/transfuser/best_model.pth