from config import GlobalConfig
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from model import TransFuser

config = GlobalConfig()
model = TransFuser(config,"cpu")

image = torch.randn(1, 3, 256, 256)
lidar = torch.randn(1, 2, 256, 256)

target_point = torch.randn(1, 2)
speed = torch.randn(1)

out = model([image], [lidar], target_point, speed)
flops, params = profile(model,([image], [lidar], target_point, speed))
flops, params = clever_format([flops, params], "%.3f")
print(f'flops: {flops}, params: {params}')