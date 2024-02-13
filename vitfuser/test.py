import torch
import torch.nn as nn

from efficientvit import EfficientViT
from tinyvit import TinyViT
from pvt import PyramidVisionTransformerV2

model = PyramidVisionTransformerV2(img_size=256, 
                patch_size=4, 
                in_chans=3, 
                num_classes=1000, 
                embed_dims=[32, 64, 160, 256],
                num_heads=[1, 2, 5, 8], 
                mlp_ratios=[4, 4, 4, 4], 
                qkv_bias=False, 
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0., 
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm,
                depths=[2, 2, 2, 2], 
                sr_ratios=[8, 4, 2, 1], 
                num_stages=4, linear=False)

print(model)

image_input = torch.randn(1, 3, 256, 256)
out = model(image_input)
print(out.shape)