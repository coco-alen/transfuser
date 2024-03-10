import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile
from thop import clever_format

from .efficientvit import EfficientViT, EfficientViT_m0, EfficientViT_m1, EfficientViT_m2, EfficientViT_m3, EfficientViT_m4, EfficientViT_m5
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
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1).contiguous()) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

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
        k = self.key(key).view(B, T_k, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
        q = self.query(query).view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
        v = self.value(value).view(B, T_k, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1).contiguous()) * (1.0 / math.sqrt(k.size(-1)))
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
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class CrossTransformerBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, q_n_embd, n_head=4, block_exp=4, attn_pdrop=.0, resid_pdrop=.0):
        super().__init__()
        self.ln_begin_kv = nn.LayerNorm(n_embd)
        self.ln_begin_q = nn.LayerNorm(q_n_embd)
        # self.ln_beforeFFN = nn.LayerNorm(n_embd)
        self.attn = CrossAttention(n_embd, q_n_embd, n_head, attn_pdrop, resid_pdrop)
        # self.mlp = nn.Sequential(
        #     nn.Linear(n_embd, block_exp * n_embd),
        #     nn.ReLU(True), # changed from GELU
        #     nn.Linear(block_exp * n_embd, n_embd),
        #     nn.Dropout(resid_pdrop),
        # )

    def forward(self, q, kv, mask=None):

        bs, c, h, w = kv.size()
        _, c_q, h_, w_ = q.size()
        assert h == h_ and w == w_, 'kv and q must have the same spatial dimensions'
        kv = kv.view(bs, c, -1).transpose(1, 2).contiguous()
        q = q.view(bs, c_q, -1).transpose(1, 2).contiguous()

        kv = self.ln_begin_kv(kv)
        q = self.ln_begin_q(q)
        
        x = kv + self.attn(q, kv, kv, mask)
        # x = x + self.mlp(self.ln_beforeFFN(x))

        x = x.transpose(1, 2).contiguous().view(bs, c, h, w)

        return x



class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.n_view = 2
        self.f_image_encoder = EfficientViT(in_chans=3, **EfficientViT_m1) # f for front
        # self.f_image_encoder = load_weight(self.f_image_encoder, torch.load('/home/yipin/program/transfuser/model_ckpt/efficientvit/efficientvit_m1.pth')["model"],strict=False)
        self.lr_image_encoder = EfficientViT(in_chans=3, **EfficientViT_m0) # lr for left and right
        # self.lr_image_encoder = load_weight(self.lr_image_encoder, torch.load('/home/yipin/program/transfuser/model_ckpt/efficientvit/efficientvit_m0.pth')["model"],strict=False)

        self.cross_attn_f1 = CrossTransformerBlock(128, 64, n_head=2)
        self.cross_attn_f2 = CrossTransformerBlock(144, 128, n_head=4)
        self.cross_attn_f3 = CrossTransformerBlock(192, 192, n_head=8)

        self.cross_attn_lr1 = CrossTransformerBlock(64, 128, n_head=2)
        self.cross_attn_lr2 = CrossTransformerBlock(128, 144, n_head=4)
        self.cross_attn_lr3 = CrossTransformerBlock(192, 192, n_head=8)


    def forward(self, f_image_list, lr_image_list):

        # assert len(f_image_list) == 2, "Should have front and focus view"
        # assert len(lr_image_list) == 2, "Should have left and right view"
        f_image_list = [self.normalize_imagenet(image_input) for image_input in f_image_list]
        lr_image_list = [self.normalize_imagenet(image_input) for image_input in lr_image_list]
        # B, C, H, W = f_image_list[0].size()
        # f_image = torch.stack(f_image_list, dim=1).view(B * self.n_view, C, H, W)
        # lr_image = torch.stack(lr_image_list, dim=1).view(B * self.n_view, C, H, W)

        f_image = torch.cat(f_image_list, dim=1)
        lr_image = torch.cat(lr_image_list, dim=1)

        f_image_features = self.f_image_encoder.patch_embed(f_image)
        lr_image_features = self.lr_image_encoder.patch_embed(lr_image)

        f_image_features = self.f_image_encoder.blocks1(f_image_features)
        lr_image_features = self.lr_image_encoder.blocks1(lr_image_features)
        f_image_features_cross = self.cross_attn_f1(lr_image_features, f_image_features)
        lr_image_features_cross = self.cross_attn_lr1(f_image_features, lr_image_features)
        f_image_features = f_image_features_cross
        lr_image_features = lr_image_features_cross

        f_image_features = self.f_image_encoder.blocks2(f_image_features)
        lr_image_features = self.lr_image_encoder.blocks2(lr_image_features)
        f_image_features_cross = self.cross_attn_f2(lr_image_features, f_image_features)
        lr_image_features_cross = self.cross_attn_lr2(f_image_features, lr_image_features)
        f_image_features = f_image_features_cross
        lr_image_features = lr_image_features_cross

        f_image_features = self.f_image_encoder.blocks3(f_image_features)
        lr_image_features = self.lr_image_encoder.blocks3(lr_image_features)
        f_image_features_cross = self.cross_attn_f3(lr_image_features, f_image_features)
        lr_image_features_cross = self.cross_attn_lr3(f_image_features, lr_image_features)
        f_image_features = f_image_features_cross
        lr_image_features = lr_image_features_cross

        f_image_features = f_image_features.flatten(2).transpose(1, 2).contiguous()
        lr_image_features = lr_image_features.flatten(2).transpose(1, 2).contiguous()
        image_features = torch.cat([f_image_features, lr_image_features], dim=1)

        return image_features

    def normalize_imagenet(self, x):
        """ Normalize input images according to ImageNet standards.
        Args:
            x (tensor): input images
        """
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

class Decoder(nn.Module):
    """ A sequence of Transformer blocks """

    def __init__(self, n_embd, depth, token_num,pred_len, n_head=4, block_exp=4, attn_pdrop=.0, resid_pdrop=.0):
        super().__init__()
        self.token_num = token_num
        self.pos_emb = nn.Parameter(torch.zeros(1,token_num + pred_len + 1, n_embd))
        self.predict_emb = nn.Parameter(torch.zeros(1, pred_len, n_embd))
        self.control_emb = nn.Parameter(torch.zeros(1, 1, n_embd))

        self.velocity_embed = nn.Linear(1, n_embd)
        self.target_embed = nn.Linear(2, n_embd)
        self.command_embed = nn.Embedding(6, n_embd)

        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop) for _ in range(depth)])

    def forward(self, image_features, target_point, velocity, command):
        bs, num, _ = image_features.size()
        assert num+3 == self.token_num, "Token number should be equal to the input feature length"

        velocity = self.velocity_embed(velocity).unsqueeze(1)
        target_point = self.target_embed(target_point).unsqueeze(1)
        command = self.command_embed(command).unsqueeze(1)
 
        predict_features = self.predict_emb.repeat(bs, 1, 1)
        control_features = self.control_emb.repeat(bs, 1, 1)
        x = torch.cat([image_features, velocity, target_point, command, predict_features, control_features], dim=1)
        x = x + self.pos_emb

        for block in self.blocks:
            x = block(x)
        
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

        embed_dim = 192
        self.encoder = Encoder(config).to(self.device)
        self.decoder = Decoder(n_embd=embed_dim,
                            depth=4,
                            token_num=35,
                            pred_len=config.pred_len,
                            n_head=4,
                            block_exp=4,
                            attn_pdrop=0.1,
                            resid_pdrop=0.1).to(self.device)
        
        self.speed_predictor = nn.Sequential(
                        nn.Linear(embed_dim, 64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64, 1),
                    ).to(self.device)
        
        self.wp_predictor = GRUWaypointsPredictor(embed_dim, waypoints=config.pred_len).to(self.device)
        self.neck = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(inplace=True),
            ).to(self.device)
        
        self.decoder_ctrl = nn.GRUCell(input_size=256+4, hidden_size=256).to(self.device)
        self.decoder_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
            ).to(self.device)
        self.policy_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.Dropout2d(p=0.5),
                nn.ReLU(inplace=True),
            ).to(self.device)
        self.dist_mu = nn.Sequential(nn.Linear(256, 2), nn.Softplus()).to(self.device)
        self.dist_sigma = nn.Sequential(nn.Linear(256, 2), nn.Softplus()).to(self.device)

        # self.brake_pred = nn.Sequential(nn.Linear(192, 2), nn.Softplus()).to(self.device)

    def forward(self, f_image_list, lr_image_list, target_point, velocity, command):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            f_image_list (list): list of front images
            lr_image_list (list): list of left and right images
            target_point (tensor): goal location registered to ego-frame
            velocity (tensor): input velocity from speedometer
        '''
        out = {} 
        image_feature = self.encoder(f_image_list, lr_image_list)
        out["pred_speed"] = self.speed_predictor(image_feature.mean(1))

        bs = image_feature.shape[0]
        dtype = image_feature.dtype
        device = image_feature.device
        fused_features = self.decoder(image_feature, target_point, velocity, command)
        predict_features = fused_features[:,-self.pred_len-1:-1,:] # predict last 4 waypoints
        # measure_features = fused_features[:,-self.pred_len-4:-self.pred_len-1,:]
        # out["brake"] = self.brake_pred(measure_features.mean(1))

        pred_wp = self.wp_predictor(predict_features, target_point)
        out["pred_wp"] = pred_wp

        # pred_wp = self.wp_predictor(predict_features)
        # out["pred_wp"] = torch.cumsum(pred_wp, 1)

        control_features = fused_features[:,-1,:]
        control_features = self.neck(control_features)
        out["pred_feature"] = control_features
        policy = self.policy_head(control_features)
        action_mu = self.dist_mu(policy)
        action_sigma = self.dist_sigma(policy)
        out["mu_branches"] = action_mu
        out["sigma_branches"] = action_sigma

        future_feature, future_mu, future_sigma = [], [], []

        # initial hidden variable to GRU
        h = torch.zeros(size=(bs, 256), dtype=dtype, device=device)
        for _ in range(self.config.pred_len):
            x_in = torch.cat([control_features, action_mu, action_sigma], dim=1)
            
            h = self.decoder_ctrl(x_in, h)
            d_feature = self.decoder_head(h)
            control_features = d_feature + control_features

            future_feature.append(control_features)

            policy = self.policy_head(control_features)
            mu = self.dist_mu(policy)
            sigma = self.dist_sigma(policy)
            
            future_mu.append(mu)
            future_sigma.append(sigma)


        out['future_feature'] = future_feature
        out['future_mu'] = future_mu
        out['future_sigma'] = future_sigma

        return out

    def process_action(self, pred):
        action = self._get_action_beta(pred['mu_branches'].view(1,2), pred['sigma_branches'].view(1,2))
        acc, steer = action.cpu().numpy()[0].astype(np.float64)
        if acc >= 0.0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.abs(acc)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)

        metadata = {
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
        }
        return steer, throttle, brake, metadata

    def _get_action_beta(self, alpha, beta):
        x = torch.zeros_like(alpha)
        x[:, 1] += 0.5
        mask1 = (alpha > 1) & (beta > 1)
        x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

        mask2 = (alpha <= 1) & (beta > 1)
        x[mask2] = 0.0

        mask3 = (alpha > 1) & (beta <= 1)
        x[mask3] = 1.0

        # mean
        mask4 = (alpha <= 1) & (beta <= 1)
        x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)

        x = x * 2 - 1

        return x

    def control_pid(self, waypoints, velocity, target):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        target = target.squeeze().data.cpu().numpy()

        # flip y (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        target[1] *= -1

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            # magnitude of vectors, used for speed
            desired_speed += np.linalg.norm(
                    waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
            if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # choice of point to aim for steering, removing outlier predictions
        # use target point if it has a smaller angle or if error is large
        # predicted point otherwise
        # (reduces noise in eg. straight roads, helps with sudden turn commands)
        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        speed = velocity[0].data.cpu().numpy()
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_4': tuple(waypoints[3].astype(np.float64)),
            'wp_3': tuple(waypoints[2].astype(np.float64)),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'angle_last': float(angle_last.astype(np.float64)),
            'angle_target': float(angle_target.astype(np.float64)),
            'angle_final': float(angle_final.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata


    def get_action(self, mu, sigma):
        action = self._get_action_beta(mu.view(1,2), sigma.view(1,2))
        acc, steer = action[:, 0], action[:, 1]
        if acc >= 0.0:
            throttle = acc
            brake = torch.zeros_like(acc)
        else:
            throttle = torch.zeros_like(acc)
            brake = torch.abs(acc)

        throttle = torch.clamp(throttle, 0, 1)
        steer = torch.clamp(steer, -1, 1)
        brake = torch.clamp(brake, 0, 1)

        return throttle, steer, brake

if __name__ == "__main__":
    from config import GlobalConfig
    config = GlobalConfig()
    model = VitFuser(config, device="cpu")

    f_image = torch.randn(1, 3, 256, 256)
    f_focus_image = torch.randn(1, 3, 256, 256)
    l_image = torch.randn(1, 3, 256, 256)
    r_image = torch.randn(1, 3, 256, 256)
    velocity = torch.randn(1)
    target_point = torch.randn(1, 2)
    command = torch.randint(0, 7, (1,))

    out = model([f_image, f_focus_image], [l_image, r_image], target_point, velocity, command)
    # for k, v in out.items():
    #     print(k, v.shape)
    from thop import profile
    from thop import clever_format

    macs, params = profile(model, inputs=([f_image, f_focus_image], [l_image, r_image], target_point, velocity, command))
    macs, params = clever_format([macs, params], "%.3f")
    print(f'macs: {macs}, params: {params}')