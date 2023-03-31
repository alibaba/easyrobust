import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math
from einops import rearrange
from .transformer_ops.transformer_function import TransformerEncoderLayer


######################################################################################
# Attention-Aware Layer
######################################################################################
class AttnAware(nn.Module):
    def __init__(self, input_nc, activation='gelu', norm='pixel', num_heads=2):
        super(AttnAware, self).__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)
        head_dim = input_nc // num_heads
        self.num_heads = num_heads
        self.input_nc = input_nc
        self.scale = head_dim ** -0.5

        self.query_conv = nn.Sequential(
            norm_layer(input_nc),
            activation_layer,
            nn.Conv2d(input_nc, input_nc, kernel_size=1)
        )
        self.key_conv = nn.Sequential(
            norm_layer(input_nc),
            activation_layer,
            nn.Conv2d(input_nc, input_nc, kernel_size=1)
        )

        self.weight = nn.Conv2d(self.num_heads*2, 2, kernel_size=1, stride=1)
        self.to_out = ResnetBlock(input_nc * 2, input_nc, 1, 0, activation, norm)

    def forward(self, x, pre=None, mask=None):
        B, C, W, H = x.size()
        q = self.query_conv(x).view(B, -1, W*H)
        k = self.key_conv(x).view(B, -1, W*H)
        v = x.view(B, -1, W*H)

        q = rearrange(q, 'b (h d) n -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b (h d) n -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b (h d) n -> b h n d', h=self.num_heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if pre is not None:
            # attention-aware weight
            B, head, N, N = dots.size()
            mask_n = mask.view(B, -1, 1, W * H).expand_as(dots)
            w_visible = (dots.detach() * mask_n).max(dim=-1, keepdim=True)[0]
            w_invisible = (dots.detach() * (1-mask_n)).max(dim=-1, keepdim=True)[0]
            weight = torch.cat([w_visible.view(B, head, W, H), w_invisible.view(B, head, W, H)], dim=1)
            weight = self.weight(weight)
            weight = F.softmax(weight, dim=1)
            # visible attention score
            pre_v = pre.view(B, -1, W*H)
            pre_v = rearrange(pre_v, 'b (h d) n -> b h n d', h=self.num_heads)
            dots_visible = torch.where(dots > 0, dots * mask_n, dots / (mask_n + 1e-8))
            attn_visible = dots_visible.softmax(dim=-1)
            context_flow = torch.einsum('bhij, bhjd->bhid', attn_visible, pre_v)
            context_flow = rearrange(context_flow, 'b h n d -> b (h d) n').view(B, -1, W, H)
            # invisible attention score
            dots_invisible = torch.where(dots > 0, dots * (1 - mask_n), dots / ((1 - mask_n) + 1e-8))
            attn_invisible = dots_invisible.softmax(dim=-1)
            self_attention = torch.einsum('bhij, bhjd->bhid', attn_invisible, v)
            self_attention = rearrange(self_attention, 'b h n d -> b (h d) n').view(B, -1, W, H)
            # out
            out = weight[:, :1, :, :]*context_flow + weight[:, 1:, :, :]*self_attention
        else:
            attn = dots.softmax(dim=-1)
            out = torch.einsum('bhij, bhjd->bhid', attn, v)

            out = rearrange(out, 'b h n d -> b (h d) n').view(B, -1, W, H)

        out = self.to_out(torch.cat([out, x], dim=1))
        return out


######################################################################################
# base modules
######################################################################################
class NoiseInjection(nn.Module):
    def __init__(self):
        super(NoiseInjection, self).__init__()

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None, mask=None):
        if noise is None:
            b, _, h, w = x.size()
            noise = x.new_empty(b, 1, h, w).normal_()
        if mask is not None:
            mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear', align_corners=True)
            return x + self.alpha * noise * (1 - mask)    # add noise only to the invisible part
        return x + self.alpha * noise


class ConstantInput(nn.Module):
    """
    add position embedding for each learned VQ word
    """
    def __init__(self, channel, size=16):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class UpSample(nn.Module):
    """ sample with convolutional operation
    :param input_nc: input channel
    :param with_conv: use convolution to refine the feature
    :param kernel_size: feature size
    :param return_mask: return mask for the confidential score
    """
    def __init__(self, input_nc, with_conv=False, kernel_size=3, return_mask=False):
        super(UpSample, self).__init__()
        self.with_conv = with_conv
        self.return_mask = return_mask
        if self.with_conv:
            self.conv = PartialConv2d(input_nc, input_nc, kernel_size=kernel_size, stride=1,
                                      padding=int(int(kernel_size-1)/2), return_mask=True)

    def forward(self, x, mask=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        mask = F.interpolate(mask, scale_factor=2, mode='bilinear', align_corners=True) if mask is not None else mask
        if self.with_conv:
            x, mask = self.conv(x, mask)
        if self.return_mask:
            return x, mask
        else:
            return x


class DownSample(nn.Module):
    """ sample with convolutional operation
        :param input_nc: input channel
        :param with_conv: use convolution to refine the feature
        :param kernel_size: feature size
        :param return_mask: return mask for the confidential score
    """
    def __init__(self, input_nc, with_conv=False, kernel_size=3, return_mask=False):
        super(DownSample, self).__init__()
        self.with_conv = with_conv
        self.return_mask = return_mask
        if self.with_conv:
            self.conv = PartialConv2d(input_nc, input_nc, kernel_size=kernel_size, stride=2,
                                      padding=int(int(kernel_size-1)/2), return_mask=True)

    def forward(self, x, mask=None):
        if self.with_conv:
            x, mask = self.conv(x, mask)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            mask = F.avg_pool2d(mask, kernel_size=2, stride=2) if mask is not None else mask
        if self.return_mask:
            return x, mask
        else:
            return x


class ResnetBlock(nn.Module):
    def __init__(self, input_nc, output_nc=None, kernel=3, dropout=0.0, activation='gelu', norm='pixel', return_mask=False):
        super(ResnetBlock, self).__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)
        self.return_mask = return_mask

        output_nc = input_nc if output_nc is None else output_nc

        self.norm1 = norm_layer(input_nc)
        self.conv1 = PartialConv2d(input_nc, output_nc, kernel_size=kernel, padding=int((kernel-1)/2), return_mask=True)
        self.norm2 = norm_layer(output_nc)
        self.conv2 = PartialConv2d(output_nc, output_nc, kernel_size=kernel, padding=int((kernel-1)/2), return_mask=True)
        self.dropout = nn.Dropout(dropout)
        self.act = activation_layer

        if input_nc != output_nc:
            self.short = PartialConv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0)
        else:
            self.short = Identity()

    def forward(self, x, mask=None):
        x_short = self.short(x)
        x, mask = self.conv1(self.act(self.norm1(x)), mask)
        x, mask = self.conv2(self.dropout(self.act(self.norm2(x))), mask)
        if self.return_mask:
            return (x + x_short) / math.sqrt(2), mask
        else:
            return (x + x_short) / math.sqrt(2)


class DiffEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, kernel_size=2, embed_dim=512, down_scale=4, num_res_blocks=2, dropout=0.0,
                 rample_with_conv=True, activation='gelu', norm='pixel', use_attn=False):
        super(DiffEncoder, self).__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)

        # start
        self.encode = PartialConv2d(input_nc, ngf, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), return_mask=True)
        # down
        self.use_attn = use_attn
        self.down_scale = down_scale
        self.num_res_blocks = num_res_blocks
        self.down = nn.ModuleList()
        out_dim = ngf
        for i in range(down_scale):
            block = nn.ModuleList()
            down = nn.Module()
            in_dim = out_dim
            out_dim = int(in_dim * 2)
            down.downsample = DownSample(in_dim, rample_with_conv, kernel_size=2, return_mask=True)
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(in_dim, out_dim, kernel_size, dropout, activation, norm, return_mask=True))
                in_dim = out_dim
            down.block = block
            self.down.append(down)
        # middle
        self.mid = nn.Module()
        self.mid.block1 = ResnetBlock(out_dim, out_dim, kernel_size, dropout, activation, norm, return_mask=True)
        if self.use_attn:
            self.mid.attn = TransformerEncoderLayer(out_dim, kernel=1)
        self.mid.block2 = ResnetBlock(out_dim, out_dim, kernel_size, dropout, activation, norm, return_mask=True)
        # end
        self.conv_out = ResnetBlock(out_dim, embed_dim, kernel_size, dropout, activation, norm, return_mask=True)

    def forward(self, x, mask=None, return_mask=False):
        x, mask = self.encode(x, mask)
        # down sampling
        for i in range(self.down_scale):
            x, mask = self.down[i].downsample(x, mask)
            for i_block in range(self.num_res_blocks):
                x, mask = self.down[i].block[i_block](x, mask)
        # middle
        x, mask = self.mid.block1(x, mask)
        if self.use_attn:
            x = self.mid.attn(x)
        x, mask = self.mid.block2(x, mask)
        # end
        x, mask = self.conv_out(x, mask)
        if return_mask:
            return x, mask
        return x


class DiffDecoder(nn.Module):
    def __init__(self, output_nc, ngf=64, kernel_size=3, embed_dim=512, up_scale=4, num_res_blocks=2, dropout=0.0, word_size=16,
                 rample_with_conv=True, activation='gelu', norm='pixel', add_noise=False, use_attn=True, use_pos=True):
        super(DiffDecoder, self).__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)
        self.up_scale = up_scale
        self.num_res_blocks = num_res_blocks
        self.add_noise = add_noise
        self.use_attn = use_attn
        self.use_pos = use_pos
        in_dim = ngf * (2 ** self.up_scale)

        # start
        if use_pos:
            self.pos_embed = ConstantInput(embed_dim, size=word_size)
        self.conv_in = PartialConv2d(embed_dim, in_dim, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2))
        # middle
        self.mid = nn.Module()
        self.mid.block1 = ResnetBlock(in_dim, in_dim, kernel_size, dropout, activation, norm)
        if self.use_attn:
            self.mid.attn = TransformerEncoderLayer(in_dim, kernel=1)
        self.mid.block2 = ResnetBlock(in_dim, in_dim, kernel_size, dropout, activation, norm)
        # up
        self.up = nn.ModuleList()
        out_dim = in_dim
        for i in range(up_scale):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            noise = nn.ModuleList()
            up = nn.Module()
            in_dim = out_dim
            out_dim = int(in_dim / 2)
            for i_block in range(num_res_blocks):
                if add_noise:
                    noise.append(NoiseInjection())
                block.append(ResnetBlock(in_dim, out_dim, kernel_size, dropout, activation, norm))
                in_dim = out_dim
                if i == 0 and self.use_attn:
                    attn.append(TransformerEncoderLayer(in_dim, kernel=1))
            up.block = block
            up.attn = attn
            up.noise = noise
            upsample = True if (i != 0) else False
            up.out = ToRGB(in_dim, output_nc, upsample, activation, norm)
            up.upsample = UpSample(in_dim, rample_with_conv, kernel_size=3)
            self.up.append(up)
        # end
        self.decode = ToRGB(in_dim, output_nc, True, activation, norm)

    def forward(self, x, mask=None):
        x = x + self.pos_embed(x) if self.use_pos else x
        x = self.conv_in(x)
        # middle
        x = self.mid.block1(x)
        if self.use_attn:
            x = self.mid.attn(x)
        x = self.mid.block2(x)
        # up
        skip = None
        for i in range(self.up_scale):
            for i_block in range(self.num_res_blocks):
                if self.add_noise:
                    x = self.up[i].noise[i_block](x, mask=mask)
                x = self.up[i].block[i_block](x)
                if len(self.up[i].attn) > 0:
                    x = self.up[i].attn[i_block](x)
            skip = self.up[i].out(x, skip)
            x = self.up[i].upsample(x)
        # end
        x = self.decode(x, skip)
        return x


class LinearEncoder(nn.Module):
    def __init__(self, input_nc, kernel_size=16, embed_dim=512):
        super(LinearEncoder, self).__init__()

        self.encode = PartialConv2d(input_nc, embed_dim, kernel_size=kernel_size, stride=kernel_size, return_mask=True)

    def forward(self, x, mask=None, return_mask=False):
        x, mask = self.encode(x, mask)
        if return_mask:
            return x, mask
        return x


class LinearDecoder(nn.Module):
    def __init__(self, output_nc, ngf=64, kernel_size=16, embed_dim=512, activation='gelu', norm='pixel'):
        super(LinearDecoder, self).__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)

        self.decode = nn.Sequential(
            norm_layer(embed_dim),
            activation_layer,
            PartialConv2d(embed_dim, ngf*kernel_size*kernel_size, kernel_size=3, padding=1),
            nn.PixelShuffle(kernel_size),
            norm_layer(ngf),
            activation_layer,
            PartialConv2d(ngf, output_nc, kernel_size=3, padding=1)
        )

    def forward(self, x, mask=None):
        x = self.decode(x)

        return torch.tanh(x)


class ToRGB(nn.Module):
    def __init__(self, input_nc, output_nc, upsample=True, activation='gelu', norm='pixel'):
        super().__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)

        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            input_nc = input_nc + output_nc

        self.conv = nn.Sequential(
            norm_layer(input_nc),
            activation_layer,
            PartialConv2d(input_nc, output_nc, kernel_size=3, padding=1)
        )

    def forward(self, input, skip=None):
        if skip is not None:
            skip = self.upsample(skip)
            input = torch.cat([input, skip], dim=1)

        out = self.conv(input)

        return torch.tanh(out)


######################################################################################
# base function for network structure
######################################################################################
def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(iter):
            lr_l = 1.0 - max(0, iter + opt.iter_count - opt.n_iter) / float(opt.n_iter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'pixel':
        norm_layer = functools.partial(PixelwiseNorm)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'relu':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'gelu':
        nonlinearity_layer = nn.GELU()
    elif activation_type == 'leakyrelu':
        nonlinearity_layer = nn.LeakyReLU(0.2)
    elif activation_type == 'prelu':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


class PixelwiseNorm(nn.Module):
    def __init__(self, input_nc):
        super(PixelwiseNorm, self).__init__()
        self.init = False
        self.alpha = nn.Parameter(torch.ones(1, input_nc, 1, 1))

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        # x = x - x.mean(dim=1, keepdim=True)
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).rsqrt()  # [N1HW]
        y = x * y  # normalize the input x volume
        return self.alpha*y


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask1 = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask1)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask1)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask / self.slide_winsize   # replace the valid value to confident score
        else:
            return output