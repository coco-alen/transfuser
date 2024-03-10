import torch
import torch.nn as nn

from efficientvit import EfficientViT
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

# print(model)

# image_input = torch.randn(1, 3, 256, 256)
# out = model(image_input)
# print(out.shape)


# model = CrossTransformerBlock(
#     n_embd = 196, 
#     q_n_embd = 128, 
#     n_head=4, 
#     block_exp=4, 
#     attn_pdrop=.0,
#     resid_pdrop=.0)

# q = torch.randn(1, 128, 16, 16)
# kv = torch.randn(1, 196, 8, 8)

# out = model(q, kv)
# print(out.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile
from thop import clever_format

model = nn.GRU(input_size=192, hidden_size=64, batch_first=True)
input = torch.randn(1, 4, 192)
emb = nn.Embedding(7, 192)

flops, params = profile(model, inputs=[input])
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)

flops, params = profile(emb, inputs=[torch.randint(0,6,[1])])
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)