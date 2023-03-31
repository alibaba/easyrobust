import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import functools
from . import base_function
from .stylegan_ops import style_function
from .transformer_ops import transformer_function


##################################################################################
# Networks
##################################################################################
def define_D(opt, img_size):
    """Create a discriminator"""
    norm_value = base_function.get_norm_layer(opt.norm)
    if 'patch' in opt.netD:
        net = NLayerDiscriminator(opt.img_nc, opt.ndf, opt.n_layers_D, norm_value, use_attn=opt.attn_D)
    elif 'style' in opt.netD:
        net = StyleDiscriminator(img_size, ndf=opt.ndf, use_attn=opt.attn_D)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % opt.netD)

    return base_function.init_net(net, opt.init_type, opt.init_gain, initialize_weights=('style' not in opt.netD))


def define_G(opt):
    """Create a decoder"""
    if 'diff' in opt.netG:
        net = base_function.DiffDecoder(opt.img_nc, opt.ngf, opt.kernel_G, opt.embed_dim, opt.n_layers_G, opt.num_res_blocks,
                                        word_size=opt.word_size, activation=opt.activation, norm=opt.norm,
                                        add_noise=opt.add_noise, use_attn=opt.attn_G, use_pos=opt.use_pos_G)
    elif 'linear' in opt.netG:
        net = base_function.LinearDecoder(opt.img_nc, opt.ngf, opt.kernel_G, opt.embed_dim, opt.activation, opt.norm)
    elif 'refine' in opt.netG:
        net = RefinedGenerator(opt.img_nc, opt.ngf, opt.embed_dim, opt.down_layers, opt.mid_layers, opt.num_res_blocks,
                               activation=opt.activation, norm=opt.norm)
    else:
        raise NotImplementedError('Decoder model name [%s] is not recognized' % opt.netG)

    return base_function.init_net(net, opt.init_type, opt.init_gain, initialize_weights=('style' not in opt.netG))


def define_E(opt):
    """Create a encoder"""
    if 'diff' in opt.netE:
        net = base_function.DiffEncoder(opt.img_nc, opt.ngf, opt.kernel_E, opt.embed_dim, opt.n_layers_G, opt.num_res_blocks,
                                        activation=opt.activation, norm=opt.norm, use_attn=opt.attn_E)
    elif 'linear' in opt.netE:
        net = base_function.LinearEncoder(opt.img_nc, opt.kernel_E, opt.embed_dim)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % opt.netE)

    return base_function.init_net(net, opt.init_type, opt.init_gain, initialize_weights=('style' not in opt.netE))


def define_T(opt):
    """Create a transformer"""
    if "original" in opt.netT:
        e_d_f = int(opt.ngf * (2 ** opt.n_layers_G))
        net = transformer_function.Transformer(e_d_f, opt.embed_dim, e_d_f, kernel=opt.kernel_T,
                    n_encoders=opt.n_encoders, n_decoders=opt.n_decoders, embed_type=opt.embed_type)
    else:
        raise NotImplementedError('Transformer model name [%s] is not recognized' % opt.netT)
    return net


##################################################################################
# Discriminator
##################################################################################
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_attn=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input examples
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)]
            if n == 2 and use_attn:
                sequence += [
                    nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=1, stride=1, bias=use_bias),
                    base_function.AttnAware(ndf * nf_mult)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class StyleDiscriminator(nn.Module):
    def __init__(self, img_size, ndf=32, blur_kernel=[1, 3, 3, 1], use_attn=False):
        super(StyleDiscriminator, self).__init__()

        channel_multiplier = ndf / 64
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: int(512 * channel_multiplier),
            64: int(256 * channel_multiplier),
            128: int(128 * channel_multiplier),
            256: int(64 * channel_multiplier),
            512: int(32 * channel_multiplier),
            1024: int(16 * channel_multiplier),
        }

        convs = [style_function.ConvLayer(3, channels[img_size], 1)]

        log_size = int(np.log2(img_size))

        in_channel = channels[img_size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i-1)]
            if i == log_size - 3 and use_attn:
                convs.append(base_function.AttnAware(in_channel))
            convs.append(style_function.StyleBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = style_function.ConvLayer(in_channel+1, channels[4], 3)
        self.final_linear = nn.Sequential(
            style_function.EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            style_function.EqualLinear(channels[4], 1),
        )

    def forward(self, x):

        out = self.convs(x)

        b, c, h, w = out.shape
        group = min(b, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, c // self.stddev_feat, h, w)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, h, w)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(b, -1)
        out = self.final_linear(out)

        return out


##################################################################################
# Generator
##################################################################################
class RefinedGenerator(nn.Module):
    def __init__(self, input_nc, ngf=64, embed_dim=512, down_layers=3, mid_layers=6, num_res_blocks=1, dropout=0.0,
                 rample_with_conv=True, activation='gelu', norm='pixel'):
        super(RefinedGenerator, self).__init__()

        activation_layer = base_function.get_nonlinearity_layer(activation)
        norm_layer = base_function.get_norm_layer(norm)
        self.down_layers = down_layers
        self.mid_layers = mid_layers
        self.num_res_blocks = num_res_blocks
        out_dims = []
        # start
        self.encode = base_function.PartialConv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1)
        # down
        self.down = nn.ModuleList()
        out_dim = ngf
        for i in range(self.down_layers):
            block = nn.ModuleList()
            down = nn.Module()
            in_dim = out_dim
            out_dims.append(out_dim)
            out_dim = min(int(in_dim * 2), embed_dim)
            down.downsample = base_function.DownSample(in_dim, rample_with_conv, kernel_size=3)
            for i_block in range(self.num_res_blocks):
                block.append(base_function.ResnetBlock(in_dim, out_dim, 3, dropout, activation, norm))
                in_dim = out_dim
            down.block = block
            self.down.append(down)
        # middle
        self.mid = nn.ModuleList()
        for i in range(self.mid_layers):
            self.mid.append(base_function.ResnetBlock(out_dim, out_dim, 3, dropout, activation, norm))
        # up
        self.up = nn.ModuleList()
        for i in range(self.down_layers):
            block = nn.ModuleList()
            up = nn.Module()
            in_dim = out_dim
            out_dim = max(out_dims[-i-1], ngf)
            for i_block in range(self.num_res_blocks):
                block.append(base_function.ResnetBlock(in_dim, out_dim, 3, dropout, activation, norm))
                in_dim = out_dim
            if i == self.down_layers - 3:
                up.attn = base_function.AttnAware(out_dim, activation, norm)
            up.block = block
            upsample = True if i != 0 else False
            up.out = base_function.ToRGB(out_dim, input_nc, upsample, activation, norm)
            up.upsample = base_function.UpSample(out_dim, rample_with_conv, kernel_size=3)
            self.up.append(up)
        # end
        self.decode = base_function.ToRGB(out_dim, input_nc, True, activation, norm)

    def forward(self, x, mask=None):
        # start
        x = self.encode(x)
        pre = None
        # down
        for i in range(self.down_layers):
            x = self.down[i].downsample(x)
            if i == 2:
                pre = x
            for i_block in range(self.num_res_blocks):
                x = self.down[i].block[i_block](x)
        # middle
        for i in range(self.mid_layers):
            x = self.mid[i](x)
        # up
        skip = None
        for i in range(self.down_layers):
            for i_block in range(self.num_res_blocks):
                x = self.up[i].block[i_block](x)
            if i == self.down_layers - 3:
                mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear', align_corners=True) if mask is not None else None
                x = self.up[i].attn(x, pre=pre, mask=mask)
            skip = self.up[i].out(x, skip)
            x = self.up[i].upsample(x)
        # end
        x = self.decode(x, skip)

        return x