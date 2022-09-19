"""
@inproceedings{mao2021discrete,
  title={Discrete Representations Strengthen Vision Transformer Robustness},
  author={Mao, Chengzhi and Jiang, Lu and Dehghani, Mostafa and Vondrick, Carl and Sukthankar, Rahul and Essa, Irfan},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
"""

from functools import partial

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import timm.models.vision_transformer
from timm.models.registry import register_model

from easyrobust.third_party.vqgan import Encoder, VectorQuantizer

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class DiscreteVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, patch_size, embed_dim, **kwargs):
        super(DiscreteVisionTransformer, self).__init__(patch_size=patch_size, embed_dim=embed_dim, **kwargs)

        self.patch_embed.proj = nn.Conv2d(3, embed_dim-256, kernel_size=patch_size, stride=patch_size)

        self.vq_encoder = Encoder(ch=128, out_ch=3, ch_mult=(1,1,2,2,4), num_res_blocks=2, attn_resolutions=(16,), dropout=0.0, in_channels=3, resolution=256, z_channels=256, double_z=False)
        self.vq_quant_conv = torch.nn.Conv2d(256, 256, 1)
        self.quantize = VectorQuantizer(1024, 256, beta=0.25)
        sd = model_zoo.load_url('http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/pretrained_models/vqgan_imagenet_f16_1024.ckpt', map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        quantize_weights, encoder_weights, quant_conv_weights = {}, {}, {}
        for k in keys:
            if 'quantize' in k:
                quantize_weights[k.replace('quantize.', '')] = sd[k]
            elif 'quant_conv' in k and 'post' not in k:
                quant_conv_weights[k.replace('quant_conv.', '')] = sd[k]
            elif 'encoder' in k:
                encoder_weights[k.replace('encoder.', '')] = sd[k]
        self.vq_encoder.load_state_dict(encoder_weights)
        self.vq_quant_conv.load_state_dict(quant_conv_weights)
        self.quantize.load_state_dict(quantize_weights)
        self.vq_encoder.eval()
        self.vq_quant_conv.eval()


    def forward_features(self, raw_x):
        
        # discrete embedding
        with torch.no_grad():
            vq_x = self.vq_quant_conv(self.vq_encoder(raw_x))
        quant = self.quantize(vq_x.detach())
        discrete_emb = quant.flatten(2).transpose(1, 2)
        
        # continous embedding
        continous_emb = self.patch_embed(raw_x)

        x = torch.cat((continous_emb, discrete_emb), dim=2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        
        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        outcome = self.norm(x)

        return outcome

    def forward(self, raw_x):
        x = self.forward_features(raw_x)
        x = self.head(x)
        return x

@register_model
def drvit_small_patch16(pretrained=False, **kwargs):
    model = DiscreteVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def drvit_base_patch16(pretrained=False, **kwargs):
    model = DiscreteVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def drvit_large_patch16(pretrained=False, **kwargs):
    model = DiscreteVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model