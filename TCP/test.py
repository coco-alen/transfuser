from efficientvit import EfficientViT, EfficientViT_m0
from model_transformer import VitFuser
from config import GlobalConfig
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
config = GlobalConfig()
image_input = torch.randn(1, 3, 256, 256).to("cpu")
image_input2 = torch.randn(1, 3, 256, 256).to("cpu")
target_point = torch.randn(1, 2).to("cpu")
speed = torch.randn(1,1).to("cpu")
target_command = torch.randint(0, 5, [1]).to("cpu")
model = VitFuser(config,"cpu")
measure = torch.randn(1, 9)
# out = model(image_input, measure, target_point)
out = model([image_input], [image_input2], target_point, speed, target_command )
flops, params = profile(model,([image_input], [image_input2], target_point, speed, target_command ))
flops, params = clever_format([flops, params], "%.3f")
print(f'flops: {flops}, params: {params}')

# warm up
# import time
# REPEAT = 1000
# for _ in range(100):
# 	pred_wp = model([image_input], [image_input2],target_point, speed, target_command )
# # speed test
# time_start = time.time()
# for _ in range(REPEAT):
# 	pred_wp = model([image_input], [image_input2],target_point, speed, target_command )
# 	torch.cuda.synchronize()
# time_end = time.time()
# print('latency: ', (time_end-time_start)/REPEAT * 1000, 'ms')

# cross = CrossAttention(192, 128, 8)
# q = torch.randn(1, 4, 128)
# kv = torch.randn(1, 8, 192)
# output = cross(q, kv, kv)
# print(output.shape)



# import torch
# from data_new import CARLA_Data
# from config import GlobalConfig

# def _get_action_beta(alpha, beta):
#     x = torch.zeros_like(alpha)
#     x[:, 1] += 0.5
#     mask1 = (alpha > 1) & (beta > 1)
#     x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

#     mask2 = (alpha <= 1) & (beta > 1)
#     x[mask2] = 0.0

#     mask3 = (alpha > 1) & (beta <= 1)
#     x[mask3] = 1.0

#     # mean
#     mask4 = (alpha <= 1) & (beta <= 1)
#     x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)

#     x = x * 2 - 1

#     return x

# def get_action(mu, sigma):
#     action = _get_action_beta(mu.view(1,2), sigma.view(1,2))
#     acc, steer = action[:, 0], action[:, 1]
#     if acc >= 0.0:
#         throttle = acc
#         brake = torch.zeros_like(acc)
#     else:
#         throttle = torch.zeros_like(acc)
#         brake = torch.abs(acc)

#     throttle = torch.clamp(throttle, 0, 1)
#     steer = torch.clamp(steer, -1, 1)
#     brake = torch.clamp(brake, 0, 1)

#     return throttle, steer, brake

# config = GlobalConfig()
# val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data,)
# data = val_set[123]
# print(data.keys())
# # print(data['action'])
# # print(data['action_mu'])
# # print(data['action_sigma'])
# print(data['target_point'])
# # print(data['target_point_aim'])


# # print(get_action(torch.tensor(data['action_mu']), torch.tensor(data['action_sigma'])))
# print(data['waypoints'])
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image
# from data_new import scale_and_crop_image

# img = Image.open("/ssd/dataset/tcp/data/tcp_carla_data/town06/routes_town06_03_30_13_34_39/rgb/0041.png")
# img = scale_and_crop_image(img, 1.0, 256, 256)
# # save image
# img.save("test1.png")
# avgpool = nn.AdaptiveAvgPool2d((32, 32))
# # conver img to tensor
# img = np.array(img)
# img = img.transpose(2, 0, 1)
# img = torch.tensor(img, dtype=torch.float32)
# img = img.unsqueeze(0)
# img_pool = avgpool(img)
# img = F.interpolate(img_pool, scale_factor=8, mode='bilinear')
# print(img.shape)

# # save image
# img = img.squeeze(0)
# img = img.numpy()
# img = img.transpose(1, 2, 0)
# img = Image.fromarray(img.astype(np.uint8))
# img.save("test2.png")