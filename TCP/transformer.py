import numpy as np
import math
import pickle
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = norm_layer(out_features, eps=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        return x



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            if isinstance(mask, float):
                mask = attn > mask
                attn = attn * mask
            else:
                attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, qkv_bias=True, dropout=0.0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=qkv_bias)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=qkv_bias)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=qkv_bias)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=qkv_bias)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # v = v.transpose(1, 2)
        
        out, attn = self.attention(q, k, v, mask=mask)
        # attn = torch.ones(sz_b, n_head, len_q, len_k, device=q.device) * (1/len_q)
        out = torch.matmul(attn, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        out = self.dropout(self.fc(out))

        out = self.layer_norm(out)

        return out


class LinAngularAttention(nn.Module):
    def __init__(
        self,
        in_dim,
        d_k,
        d_v,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
    ):
        super().__init__()
        assert in_dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads

        self.d_k = d_k
        self.d_v = d_v

        self.scale = d_k**-0.5
        self.sparse_reg = sparse_reg

        self.w_qs = nn.Linear(in_dim, num_heads * d_k, bias=qkv_bias)
        self.w_ks = nn.Linear(in_dim, num_heads * d_k, bias=qkv_bias)
        self.w_vs = nn.Linear(in_dim, num_heads * d_v, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, q, k, v, mask=None):
        assert q.shape == k.shape and k.shape == v.shape, "input shape must be equal"
        N, L, C = q.shape

        d_k, d_v, n_head = self.d_k, self.d_v, self.num_heads
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, n_head, len_q, d_k)
        k = self.w_ks(k).view(sz_b, n_head, len_k, d_k)
        v = self.w_vs(v).view(sz_b, n_head, len_v, d_v)

        if self.sparse_reg:
            attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v)

        attn = torch.matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                torch.matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * torch.matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * torch.matmul(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim = None,
        num_heads = 8,
        ffn_expand_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mask_limlt = None,
    ):
        super().__init__()

        # self.norm = norm_layer(in_dim)
        self.attn = MultiHeadAttention(
            n_head = num_heads, 
            d_model = in_dim,
            d_k = in_dim // num_heads,
            d_v = in_dim // num_heads,
            qkv_bias = qkv_bias,
            dropout=0.0)

        ffn_hidden_dim = int(in_dim * ffn_expand_ratio)
        out_dim = out_dim or in_dim

        self.ffn = FFN(
            in_features=in_dim,
            hidden_features=ffn_hidden_dim,
            out_features=out_dim,
            act_layer=act_layer,
            drop=drop,
            norm_layer = norm_layer,
        )
        self.mask_limlt = mask_limlt

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.attn(x,x,x, self.mask_limlt)
        x = self.ffn(x)
        return x
