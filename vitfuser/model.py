import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from pvt import OverlapPatchEmbed, Block, CrossAttention

from config import GlobalConfig
from thop import profile
from thop import clever_format

class pvt_layer(nn.Module):
    def __init__(self, 
                 img_size: int,
                 patch_in_chans: int,
                 embed_dims,
                 stage_idx: int,
                 drop_path_rate: list,
                 cross: bool = False,
                 cross_patch_chans: int = 0,
                 num_heads: int = 4,
                 mlp_ratios: int = 4,
                 depths: int = 2,
                 sr_ratios: int = 8,
                 qkv_bias=False, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., norm_layer=nn.LayerNorm, linear=False):

        super(pvt_layer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(img_size=img_size if stage_idx == 0 else img_size // (2 ** (stage_idx + 1)),
                                        patch_size=7 if stage_idx == 0 else 3,
                                        stride=4 if stage_idx == 0 else 2,
                                        in_chans=patch_in_chans,
                                        embed_dim=embed_dims)
        self.cross = cross
        if self.cross == True:
            self.cross_patch = OverlapPatchEmbed(img_size=img_size if stage_idx == 0 else img_size // (2 ** (stage_idx + 1)),
                                        patch_size=7 if stage_idx == 0 else 3,
                                        stride=4 if stage_idx == 0 else 2,
                                        in_chans=cross_patch_chans,
                                        embed_dim=embed_dims)
            self.cross_attn = CrossAttention(
                n_embd=embed_dims,
                n_head=num_heads,
                attn_pdrop=attn_drop_rate,
                resid_pdrop=drop_rate,
            )
            self.vel_emb = nn.Linear(1, embed_dims)

        self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate[j], norm_layer=norm_layer,
            sr_ratio=sr_ratios, linear=linear)
            for j in range(depths)])
        self.norm = norm_layer(embed_dims)

    def forward(self, kv, q, velocity):
        B = kv.shape[0]
        x, H, W = self.patch_embed(kv)
        if self.cross == True:
            q, H_, W_ = self.cross_patch(q)
            velc = self.vel_emb(velocity.unsqueeze(1)).unsqueeze(1)
            q = q + velc
            assert H == H_ and W == W_, "kv and q must have the same spatial resolution!"
            x = self.cross_attn(q, x, x)

        for blk in self.block:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


def create_pvt_layers(
    img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512], q_embed_dims=[32, 64, 160, 256],
    num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
    depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False
):

    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
    cur = 0

    layers = []
    for i in range(num_stages):
        layer = pvt_layer(
            img_size=img_size,
            patch_in_chans=in_chans if i == 0 else embed_dims[i - 1],
            embed_dims=embed_dims[i],
            stage_idx=i,
            cross=True if i > 0 else False,
            cross_patch_chans=in_chans if i == 0 else q_embed_dims[i - 1],
            drop_path_rate=dpr[cur:cur + depths[i]],
            num_heads=num_heads[i],
            mlp_ratios=mlp_ratios[i],
            depths=depths[i],
            sr_ratios=sr_ratios[i],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer, linear=linear
        )
        cur += depths[i]
        layers.append(layer)
    return nn.ModuleList(layers)

class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.image_encoder = create_pvt_layers(
                img_size=config.input_resolution,
                in_chans=3,
                embed_dims=[32, 64, 160, 256], 
                q_embed_dims=[16, 32, 96, 256],
                num_heads=[1, 2, 5, 8], 
                mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                drop_rate = config.resid_pdrop,
                attn_drop_rate = config.attn_pdrop,)
        
        self.lidar_encoder = create_pvt_layers(
                img_size=config.input_resolution,
                in_chans=2,
                embed_dims=[16, 32, 96, 256], 
                q_embed_dims=[32, 64, 160, 256], 
                num_heads=[1, 2, 4, 4], 
                mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                depths=[1, 1, 2, 2], sr_ratios=[4, 4, 2, 1],
                drop_rate = config.resid_pdrop,
                attn_drop_rate = config.attn_pdrop,)
        self.avp = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, image_list, lidar_list, velocity, interaction="sum"):

        assert len(image_list) == 1, "Only single image input supported"
        assert len(lidar_list) == 1, "Only single lidar input supported"

        image = image_list[0]
        lidar = lidar_list[0]

        for i in range(len(self.image_encoder)):
            image_next = self.image_encoder[i](image, lidar, velocity)
            lidar_next = self.lidar_encoder[i](lidar, image, velocity)
            image = image_next
            lidar = lidar_next
        
        image = self.avp(image).flatten(1)
        lidar = self.avp(lidar).flatten(1)

        if interaction == "sum":
            out = image + lidar
        elif interaction == "concat":
            out = torch.cat(image, lidar, dim=1)
        elif interaction == "dot":
            T = torch.cat([image.unsqueeze(1), lidar.unsqueeze(1)], dim=1)
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            batch_size, ni, nj = Z.shape
            li, lj = torch.tril_indices(ni, nj, offset=0)
            Zflat = Z[:, li, lj]
            out = torch.cat([image, Zflat], dim=1)
        else:
            raise ValueError(f"Unknown interaction type {interaction}")

        return out

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

        self.norm = nn.LayerNorm(256).to(self.device)
        self.join = nn.Sequential(
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)
        self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
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
        fused_features = self.encoder(image_list, lidar_list, velocity, "sum")
        fused_features = self.norm(fused_features)
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

# config = GlobalConfig()
# model = TransFuser(config,"cpu")

# print(model)
# image = torch.randn(10, 3, 256, 256)
# lidar = torch.randn(10, 2, 256, 256)
# velocity = torch.randn(10)
# target_point = torch.randn(10, 2)
# out = model([image], [lidar], target_point, velocity)
# print(out.shape)
# flops, params = profile(model, inputs=([image], [lidar],target_point, velocity))
# flops, params = clever_format([flops, params], "%.3f")
# print(f'flops: {flops}, params: {params}')