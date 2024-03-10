import os
import time

import torch
import torch.nn as nn
from thop import profile
from thop import clever_format

from efficientvit import EfficientViT, EfficientViT_m0, EfficientViT_m1, EfficientViT_m2, EfficientViT_m3, EfficientViT_m4, EfficientViT_m5
from tinyvit import TinyViT
from pvt import PyramidVisionTransformerV2
from model import CrossTransformerBlock

# model = PyramidVisionTransformerV2(img_size=256, 
#                 patch_size=4, 
#                 in_chans=3, 
#                 num_classes=1000, 
#                 embed_dims=[32, 64, 160, 256],
#                 num_heads=[1, 2, 5, 8], 
#                 mlp_ratios=[4, 4, 4, 4], 
#                 qkv_bias=False, 
#                 qk_scale=None,
#                 drop_rate=0.,
#                 attn_drop_rate=0., 
#                 drop_path_rate=0., 
#                 norm_layer=nn.LayerNorm,
#                 depths=[2, 2, 2, 2], 
#                 sr_ratios=[8, 4, 2, 1], 
#                 num_stages=4, linear=False)


# model = EfficientViT(img_size=256,
#                  patch_size=16,
#                  in_chans=3,
#                  stages=['s', 's', 's'],
#                  embed_dim=[64, 128, 192],
#                  key_dim=[16, 16, 16],
#                  depth=[1, 2, 3],
#                  num_heads=[4, 4, 4],
#                  window_size=[7, 7, 7],
#                  kernels=[5, 5, 5, 5],
#                  down_ops=[['subsample', 2], ['subsample', 2], ['']])
model = EfficientViT(**EfficientViT_m5)
print(model)
REPEAT = 1000

image_input = torch.randn(1, 3, 256, 256)
feature = model.patch_embed(image_input)
print(feature.shape)
feature = model.blocks1(feature)
print(feature.shape)
feature = model.blocks2(feature)
print(feature.shape)
feature = model.blocks3(feature)
print(feature.shape)
# flops, params = profile(model, inputs=[image_input])
# flops, params = clever_format([flops, params], "%.3f")
# print(f'flops: {flops}, params: {params}')
# # warm up
# for _ in range(100):
# 	pred_wp = model(image_input)
# # speed test
# time_start = time.time()
# for _ in range(REPEAT):
# 	pred_wp = model(image_input)
# 	torch.cuda.synchronize()
# time_end = time.time()
# print('latency: ', (time_end-time_start)/REPEAT * 1000, 'ms')
