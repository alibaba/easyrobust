# https://github.com/whai362/PVT
import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

import torch

from util.misc import NestedTensor
from typing import Dict, List
from .backbone import build_position_encoding, Joiner


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # x = x.transpose(1, 2).view(B, C, H, W)
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


class Conv1x1(nn.Module):
    def __init__(self, in_channel, out_channel, force=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(out_channel)
        if not force and in_channel == out_channel:
            self.conv = self.norm = nn.Identity()

    def forward(self, x, H=None, W=None):
        # x: (b, n, c) or (b, c, h, w)
        # ret: (b, n, c)
        if len(x.shape) == 3:
            B = x.shape[0]
            x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Conv3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channel)

    def forward(self, x, H=None, W=None):
        # x: (b, n, c) or (b, c, h, w)
        # ret: (b, n, c)
        if len(x.shape) == 3:
            B = x.shape[0]
            x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.conv(x)
        newshape = tuple(x.shape[-2:])
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, newshape


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False, poolsize=7):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(poolsize)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FuseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False, poolsize=7):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            self.sr = nn.ModuleList([ nn.Conv2d(dim, dim, kernel_size=s, stride=s) for s in sr_ratio ])
            self.norm = nn.ModuleList([ nn.LayerNorm(dim) for _ in sr_ratio ])
        else:
            self.pool = nn.AdaptiveAvgPool2d(poolsize)
            self.sr = nn.ModuleList([ nn.Conv2d(dim, dim, kernel_size=1, stride=1) for _ in sr_ratio ])
            self.norm = nn.ModuleList([ nn.LayerNorm(dim) for _ in sr_ratio ])
            self.act = nn.GELU()
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward(self, out, shapes):
        Q = []
        for x in out:
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            Q.append(q)

        K = []
        V = []
        assert len(out) == len(shapes) == len(self.sr) == len(self.norm)
        for x, shape, sr, norm in zip(out, shapes, self.sr, self.norm):
            H, W = shape
            x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            if self.linear:
                x = self.pool(x)
            x = sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = norm(x)
            if self.linear:
                x = self.act(x)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            K.append(k)
            V.append(v)

        Q = torch.cat(Q, dim=2)
        K = torch.cat(K, dim=2)
        V = torch.cat(V, dim=2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        X = (attn @ V).transpose(1, 2).reshape(B, -1, C)
        X = self.proj(X)
        X = self.proj_drop(X)

        return X

    def forward_extra(self, out_q, shapes_q, out_kv, shapes_kv):
        Q = []
        for x in out_q:
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            Q.append(q)

        K = []
        V = []
        assert len(out_kv) == len(shapes_kv) == len(self.sr) == len(self.norm)
        for x, shape, sr, norm in zip(out_kv, shapes_kv, self.sr, self.norm):
            H, W = shape
            x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            if self.linear:
                x = self.pool(x)
            x = sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = norm(x)
            if self.linear:
                x = self.act(x)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            K.append(k)
            V.append(v)

        Q = torch.cat(Q, dim=2)
        K = torch.cat(K, dim=2)
        V = torch.cat(V, dim=2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        X = (attn @ V).transpose(1, 2).reshape(B, -1, C)
        X = self.proj(X)
        X = self.proj_drop(X)

        return X


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False, poolsize=7,
                 fuse=False, fuse_extra=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sr_ratio = tuple(sr_ratio) if isinstance(sr_ratio, (list, tuple)) else sr_ratio
        self.fuse = fuse
        self.fuse_extra = fuse_extra
        if not fuse:
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear, poolsize=poolsize)
        else:
            self.attn = FuseAttention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear, poolsize=poolsize)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward(self, x, H, W, shapes=None, extra_kv=None, extra_kv_shapes=None):
        if self.fuse:
            if self.fuse_extra:
                sizes = [H * W for H, W in shapes]
                
                x = self.norm1(x)

                q = list(x.split(sizes, dim=1))
                q_shapes = list(shapes)
                kv = extra_kv + q
                kv_shapes = extra_kv_shapes + q_shapes
                x = x + self.drop_path(self.attn.forward_extra(q, q_shapes, kv, kv_shapes))

                x = self.norm2(x)

                x = [self.mlp(x_, H, W) for x_, (H, W) in zip(x.split(sizes, dim=1), shapes)]
                x = torch.cat(x, dim=1)

                x = x + self.drop_path(x)

            else:
                sizes = [H * W for H, W in shapes]
                
                x = self.norm1(x)

                x = x + self.drop_path(self.attn(x.split(sizes, dim=1), shapes))

                x = self.norm2(x)

                x = [self.mlp(x_, H, W) for x_, (H, W) in zip(x.split(sizes, dim=1), shapes)]
                x = torch.cat(x, dim=1)

                x = x + self.drop_path(x)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
                depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False,
                # fusing module
                fuse_mode=False, fuse_dim=256, fuse_num_heads=8, fuse_mlp_ratios=4, fuse_depth=3, fuse_linear=False, fuse_start_lvl=1, fuse_num_addition=1,
                fuse_dense_lookback=False, fuse_lookback_extra_depth=1, fuse_single_scale=False
                ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # cross scale fussion
        self.fuse_mode = fuse_mode
        self.fuse_dense_lookback = fuse_dense_lookback
        self.fuse_lookback_extra_depth = fuse_lookback_extra_depth
        self.fuse_start_lvl = fuse_start_lvl
        self.fuse_num_addition = fuse_num_addition
        self.fuse_single_scale = fuse_single_scale
        if fuse_mode:
            # proj
            self.fuse_proj = nn.ModuleList([ Conv1x1(d, fuse_dim) for d in embed_dims[fuse_start_lvl:] ])

            # addition scale
            in_channels = embed_dims[-1]
            self.fuse_addin_proj = []
            for _ in range(fuse_num_addition):
                self.fuse_addin_proj.append(Conv3x3(in_channels, fuse_dim))
                in_channels = fuse_dim
            self.fuse_addin_proj = nn.ModuleList(self.fuse_addin_proj)

            # dense fusing
            fuse_num_scale = num_stages - fuse_start_lvl + fuse_num_addition

            if isinstance(fuse_depth, int):
                assert fuse_depth % (fuse_num_scale - 1) == 0
                fuse_depth_per_scale = [fuse_depth // (fuse_num_scale - 1)] * (fuse_num_scale - 1)
                dpr = [x.item() for x in torch.linspace(0, drop_path_rate, fuse_depth)]
            else:
                assert isinstance(fuse_depth, (list, tuple))
                assert len(fuse_depth) == fuse_num_scale - 1
                fuse_depth_per_scale = fuse_depth
                dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(fuse_depth))]
            sr_ratios = list(sr_ratios + [sr_ratios[-1]] * fuse_num_addition)
            for k in range(fuse_num_scale - 1):  # no need to fuse if you have only one scale
                srr = sr_ratios[fuse_start_lvl : fuse_start_lvl + k + 2]  # not nearby or nearby_extra kv: all exist scales
                fuse_block = nn.ModuleList([Block(
                    dim=fuse_dim, num_heads=fuse_num_heads, mlp_ratio=fuse_mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    sr_ratio=srr, linear=fuse_linear, fuse=True, fuse_extra=fuse_dense_lookback)
                    for i in range(fuse_depth_per_scale[k])])

                fuse_norm = nn.LayerNorm(fuse_dim)

                setattr(self, f'fuse_block{k+2}', fuse_block)
                setattr(self, f'fuse_norm{k+2}', fuse_norm)

            if fuse_dense_lookback and fuse_lookback_extra_depth > 0:
                fuse_block = nn.ModuleList([Block(
                    dim=fuse_dim, num_heads=fuse_num_heads, mlp_ratio=fuse_mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[fuse_start_lvl], linear=fuse_linear)
                    for i in range(fuse_lookback_extra_depth)])

                fuse_norm = nn.LayerNorm(fuse_dim)
                print('lookback srr:', sr_ratios[fuse_start_lvl])

                setattr(self, f'fuse_block1', fuse_block)
                setattr(self, f'fuse_norm1', fuse_norm)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        outs = outs[self.scale_start:]

        return outs

    def forward_dense(self, x):
        B = x.shape[0]
        outs = []
        spatial_shapes = []

        for i in range(self.num_stages + self.fuse_num_addition):
            if i < self.num_stages:
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                block = getattr(self, f"block{i + 1}")
                norm = getattr(self, f"norm{i + 1}")

                x, H, W = patch_embed(x)
                for blk in block:
                    x = blk(x, H, W)
                
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            else:  # additional conv
                addin_proj = self.fuse_addin_proj[i - self.num_stages]
                x, (H, W) = addin_proj(x)
                x = x.transpose(1, 2).reshape(B, -1, H, W)

            if i >= self.fuse_start_lvl:
                # proj to fuse_dim
                if i < self.num_stages:
                    fuse_proj = self.fuse_proj[i - self.fuse_start_lvl]
                    outs.append(fuse_proj(x))
                    spatial_shapes.append((H, W))
                else:
                    outs.append(x.flatten(2).transpose(1, 2))
                    spatial_shapes.append((H, W))

                if len(outs) == 1 and self.fuse_dense_lookback and self.fuse_lookback_extra_depth > 0:
                    fuse_block = getattr(self, f"fuse_block1")
                    fuse_norm = getattr(self, f"fuse_norm1")
                    outs = outs[0]
                    for blk in fuse_block:
                        outs = blk(outs, *spatial_shapes[0])
                    outs = fuse_norm(outs)
                    outs = [outs]

                if len(outs) >= 2:
                    fuse_block = getattr(self, f"fuse_block{len(outs)}")
                    fuse_norm = getattr(self, f"fuse_norm{len(outs)}")
                    if self.fuse_dense_lookback:
                        outs_holdout, spatial_shapes_holdout = outs[:-1], spatial_shapes[:-1]
                        outs, spatial_shapes = outs[-1:], spatial_shapes[-1:]   
                    outs = torch.cat(outs, 1)
                    for blk in fuse_block:
                        if self.fuse_dense_lookback:
                            outs = blk(outs, None, None, spatial_shapes, outs_holdout, spatial_shapes_holdout)
                        else:
                            outs = blk(outs, None, None, spatial_shapes)
                    outs = fuse_norm(outs)
                    
                    sizes = [h * w for h, w in spatial_shapes]
                    outs = list(outs.split(sizes, dim=1))

                    if self.fuse_dense_lookback:
                        outs = outs_holdout + outs
                        spatial_shapes = spatial_shapes_holdout + spatial_shapes

        outs = [x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x, (H, W) in zip(outs, spatial_shapes)]
        return outs

    def forward(self, tensor_list: NestedTensor):
        if self.fuse_mode:
            xs = self.forward_dense(tensor_list.tensors)
        else:
            xs = self.forward_features(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for i, x in enumerate(xs):
            if self.fuse_single_scale and i < len(xs) - 1: continue
            name = str(i)
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


def load_weight(model, state_dict_all):
    def intersect_dicts(da, db, exclude=()):
        # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
        return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}
    state_dict = intersect_dicts(state_dict_all, model.state_dict())  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    print('Transferred %g/%g items' % (len(state_dict), len(model.state_dict())))
    return model
    

def build_pvt(args):
    if args.pvt == 'pvt_v2_b2_li':
        config = dict(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True)
        backbone = PyramidVisionTransformerV2(**config)
        backbone.strides = [8, 16, 32]
        backbone.num_channels = [128, 320, 512]
        backbone.scale_start = 1
    elif args.pvt == 'def-ddetr':
        assert args.no_encoder and args.no_input_proj
        config = dict(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True, fuse_mode=True, fuse_mlp_ratios=4, fuse_depth=[3,3,3], fuse_linear=True, 
            fuse_dense_lookback=True, fuse_lookback_extra_depth=0)
        backbone = PyramidVisionTransformerV2(**config)
        backbone.strides = [8, 16, 32, 64]
        backbone.num_channels = [256,256,256,256]
        backbone.non_backbone_names = sorted(list(set(['backbone.0.' + name.split('.')[0] 
                                            for name, param in backbone.named_parameters() if param.requires_grad and 'fuse' in name])))
        backbone.backbone_names     = sorted(list(set(['backbone.0.' + name.split('.')[0] 
                                            for name, param in backbone.named_parameters() if param.requires_grad and 'fuse' not in name])))
    elif args.pvt == 'ddetr':
        assert args.no_encoder and args.no_input_proj and args.num_feature_levels == 1
        config = dict(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True, fuse_mode=True, fuse_mlp_ratios=4, fuse_depth=[3,3], fuse_linear=True, 
            fuse_dense_lookback=True, fuse_lookback_extra_depth=0, fuse_num_addition=0, fuse_single_scale=True)
        backbone = PyramidVisionTransformerV2(**config)
        backbone.strides = [32]
        backbone.num_channels = [256]
        backbone.non_backbone_names = sorted(list(set(['backbone.0.' + name.split('.')[0] 
                                            for name, param in backbone.named_parameters() if param.requires_grad and 'fuse' in name])))
        backbone.backbone_names     = sorted(list(set(['backbone.0.' + name.split('.')[0] 
                                            for name, param in backbone.named_parameters() if param.requires_grad and 'fuse' not in name])))
    else:
        raise NotImplementedError
    print('*'*30)
    print('PVTv2 config:', config)
    print('*'*30)

    if args.pvt_resume:
        ckpt = torch.load(args.pvt_resume, map_location="cpu")
        if 'model' in ckpt:
            ckpt = ckpt['model']
        load_weight(backbone, ckpt)

    position_embedding = build_position_encoding(args)
    model = Joiner(backbone, position_embedding)
    return model