import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile
from thop import clever_format

from .efficientvit import EfficientViT, EfficientViT_m0, EfficientViT_m1, EfficientViT_m2, EfficientViT_m3, replace_batchnorm
from .utils import load_weight

torch.autograd.set_detect_anomaly(True)

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class CrossAttention(nn.Module):
    """
    A vanilla multi-head masked cross-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, q_n_embd = None, n_head=4, attn_pdrop=.0, resid_pdrop=.0):
        super().__init__()
        if q_n_embd is None:
            q_n_embd = n_embd
        assert n_embd % n_head == 0, 'Feature length must be divisible by n_head'
        assert q_n_embd % n_head == 0, 'Query feature length must be divisible by n_head'
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(q_n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, query, key, value, mask=None):
        assert query.size(0) == key.size(0) == value.size(0), 'Batch_size dimension of query, key, value must be the same'
        assert key.size(1) == value.size(1), 'SeqLen dimension of key, value must be the same'
        B, T_q, _ = query.size()
        B, T_k, C = key.size()


        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, T_k, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(query).view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(value).view(B, T_k, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = torch.masked_fill(att, mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class TransformerBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embd, n_head=4, block_exp=4, attn_pdrop=.0, resid_pdrop=.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CrossTransformerBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, q_n_embd, n_head=4, block_exp=4, attn_pdrop=.0, resid_pdrop=.0):
        super().__init__()
        self.ln_begin_kv = nn.LayerNorm(n_embd)
        self.ln_begin_q = nn.LayerNorm(q_n_embd)
        self.ln_beforeFFN = nn.LayerNorm(n_embd)
        self.attn = CrossAttention(n_embd, q_n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, q, kv, mask=None):

        bs, c, h, w = kv.size()
        _, c_q, h_, w_ = q.size()
        assert h == h_ and w == w_, 'kv and q must have the same spatial dimensions'
        kv = kv.view(bs, c, -1).transpose(1, 2)
        q = q.view(bs, c_q, -1).transpose(1, 2)

        kv = self.ln_begin_kv(kv)
        q = self.ln_begin_q(q)
        
        x = kv + self.attn(q, kv, kv, mask)
        x = x + self.mlp(self.ln_beforeFFN(x))

        x = x.transpose(1, 2).view(bs, c, h, w)

        return x



class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.image_encoder = EfficientViT(**EfficientViT_m1)
        # self.image_encoder = load_weight(self.image_encoder, torch.load('/home/gyp/program/my_transfuser/transfuser/model_ckpt/efficientvit/efficientvit_m1.pth')["model"],strict=False)
        self.lidar_encoder = EfficientViT(in_chans=2, **EfficientViT_m0)
        # self.lidar_encoder = load_weight(self.lidar_encoder, torch.load('/home/gyp/program/my_transfuser/transfuser/model_ckpt/efficientvit/efficientvit_m0.pth')["model"],strict=False)

        self.cross_attn_image1 = CrossTransformerBlock(128, 64, n_head=2)
        self.cross_attn_image2 = CrossTransformerBlock(144, 128, n_head=4)
        self.cross_attn_image3 = CrossTransformerBlock(192, 192, n_head=4)

        self.cross_attn_lidar1 = CrossTransformerBlock(64, 128, n_head=2)
        self.cross_attn_lidar2 = CrossTransformerBlock(128, 144, n_head=4)
        self.cross_attn_lidar3 = CrossTransformerBlock(192, 192, n_head=4)


    def forward(self, image_list, lidar_list):

        assert len(image_list) == 1, "Only single image input supported"
        assert len(lidar_list) == 1, "Only single lidar input supported"
        image = image_list[0]
        lidar = lidar_list[0]

        image_features = self.image_encoder.patch_embed(image)
        lidar_features = self.lidar_encoder.patch_embed(lidar)

        image_features = self.image_encoder.blocks1(image_features)
        lidar_features = self.lidar_encoder.blocks1(lidar_features)
        image_features_cross = self.cross_attn_image1(lidar_features, image_features)
        lidar_features_cross = self.cross_attn_lidar1(image_features, lidar_features)
        image_features = image_features_cross
        lidar_features = lidar_features_cross

        image_features = self.image_encoder.blocks2(image_features)
        lidar_features = self.lidar_encoder.blocks2(lidar_features)
        image_features_cross = self.cross_attn_image2(lidar_features, image_features)
        lidar_features_cross = self.cross_attn_lidar2(image_features, lidar_features)
        image_features = image_features_cross
        lidar_features = lidar_features_cross

        image_features = self.image_encoder.blocks3(image_features)
        lidar_features = self.lidar_encoder.blocks3(lidar_features)
        image_features_cross = self.cross_attn_image3(lidar_features, image_features)
        lidar_features_cross = self.cross_attn_lidar3(image_features, lidar_features)
        image_features = image_features_cross
        lidar_features = lidar_features_cross


        return image_features, lidar_features
    
class Fuser(nn.Module):
    """ A sequence of Transformer blocks """

    def __init__(self, n_embd, depth, token_num, n_head=4, block_exp=4, attn_pdrop=.0, resid_pdrop=.0):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1,token_num, n_embd))

        self.velocity_embed = nn.Linear(1, n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop) for _ in range(depth)])

    def forward(self, image_features, lidar_features, velocity):
        bs, c, h, w = image_features.size()
        image_features = image_features.view(bs, c, -1).transpose(1, 2)
        lidar_features = lidar_features.view(bs, c, -1).transpose(1, 2)
        velocity = self.velocity_embed(velocity.unsqueeze(1)).unsqueeze(1)
        x = torch.cat([image_features, lidar_features, velocity], dim=1)
        x = x + self.pos_emb

        for block in self.blocks:
            x = block(x)
        
        tokenNum = h * w
        image_features = x[:, :tokenNum, :].mean(dim=1)
        lidar_features = x[:, tokenNum:2*tokenNum, :].mean(dim=1)
        measure_features = x[:, 2*tokenNum:, :].mean(dim=1)
        x = torch.cat([image_features, lidar_features, measure_features], dim=1)
        return x

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class GRUWaypointsPredictor(nn.Module):
    def __init__(self, input_dim, waypoints=10):
        super().__init__()
        # self.gru = torch.nn.GRUCell(input_size=input_dim, hidden_size=64)
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=64, batch_first=True)
        self.encoder = nn.Linear(2, 64)
        self.decoder = nn.Linear(64, 2)
        self.waypoints = waypoints

    def forward(self, x, target_point):
        bs = x.shape[0]
        z = self.encoder(target_point).unsqueeze(0)
        output, _ = self.gru(x, z)
        output = output.reshape(bs * self.waypoints, -1)
        output = self.decoder(output).reshape(bs, self.waypoints, 2)
        output = torch.cumsum(output, 1)
        return output


class VitFuser(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len

        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        self.encoder = Encoder(config).to(self.device)
        self.fuser = Fuser(n_embd=192,
                            depth=4,
                            token_num=33,
                            n_head=4,
                            block_exp=4,
                            attn_pdrop=0.0,
                            resid_pdrop=0.0).to(self.device)

        # self.norm = nn.LayerNorm(512).to(self.device)
        self.join = nn.Sequential(
                            nn.Linear(576, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)
        # self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
        self.decoder = GRUWaypointsPredictor(64, waypoints=self.pred_len).to(self.device)
        self.output = nn.Linear(64, 2).to(self.device)
        
    def forward(self, image_list, lidar_list, target_point, velocity):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            velocity (tensor): input velocity from speedometer
        '''
        image_feature, lidar_feature = self.encoder(image_list, lidar_list)
        fused_features = self.fuser(image_feature, lidar_feature, velocity)
        z = self.join(fused_features)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)
        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            # x_in = torch.cat([x, target_point], dim=1)
            x_in = x + target_point
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp

    def control_pid(self, waypoints, velocity):
        ''' 
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if(speed < 0.01):
            angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata

model = GRUWaypointsPredictor(64, waypoints=4)
print(model)
image = torch.randn(1, 64)
target = torch.randn(1,2)
out = model(image, target)
print(out.shape)
# from config import GlobalConfig
# config = GlobalConfig()

# image = torch.randn(1, 3, 256, 256)
# lidar = torch.randn(1, 2, 256, 256)
# velocity = torch.randn(1)
# target_point = torch.randn(1,2)


# model = Encoder(config)
# print(model)
# out = model([image], [lidar])
# print(out[0].shape, out[1].shape)
# flops, params = profile(model, inputs=([image], [lidar]))
# flops, params = clever_format([flops, params], "%.3f")
# print(f'flops: {flops}, params: {params}')

# model = Fuser(n_embd=192,
#                 depth=4,
#                 n_head=4,
#                 block_exp=4,
#                 attn_pdrop=0.0,
#                 resid_pdrop=0.0)
# feature = model(out[0], out[1], veloc)
# print(feature.shape)
# flops, params = profile(model, inputs=(out[0], out[1], veloc))
# flops, params = clever_format([flops, params], "%.3f")
# print(f'flops: {flops}, params: {params}')

# model = VitFuser(config, 'cpu')
# print(model)
# out = model([image], [lidar], target_point, velocity)
# print(out.shape)
# flops, params = profile(model, inputs=([image], [lidar], target_point, velocity))
# flops, params = clever_format([flops, params], "%.3f")
# print(f'flops: {flops}, params: {params}')

# import time
# REPEAT = 1000
# for _ in range(100):
# 	pred_wp = model([image], [lidar], target_point, velocity)
# # speed test
# time_start = time.time()
# for _ in range(REPEAT):
# 	pred_wp = model([image], [lidar], target_point, velocity)
# 	torch.cuda.synchronize()
# time_end = time.time()
# print('latency: ', (time_end-time_start)/REPEAT * 1000, 'ms')