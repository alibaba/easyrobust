"""
2D Vision Transformer class with convolution layer.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in DETR
    * decoder returns a stack of activations from all encoding layers
"""
import copy
import torch
from torch import nn
from einops import rearrange
from .. import base_function
from .position_embedding import build_position_embed


######################################################################################
# Transformer
######################################################################################
class VQTransformer(nn.Module):
    def __init__(self, embed_dim, num_embeds, dim_conv=2048, kernel=3, num_heads=8, n_encoders=12,
                 n_decoders=12, dropout=0., activation='gelu', norm='pixel', embed_type='learned'):
        super(VQTransformer, self).__init__()

        norm_layer = base_function.get_norm_layer(norm)
        activation_layer = base_function.get_nonlinearity_layer(activation)
        self.token_embed = nn.Embedding(num_embeds, embed_dim)
        self.pos_embed = build_position_embed(embed_type=embed_type, feats_dim=embed_dim)
        self.drop = nn.Dropout(dropout)
        self.encoder_trans = TransformerEncoder(embed_dim, num_heads, n_encoders, dim_conv, kernel, dropout, activation, norm)
        self.decoder_trans = TransformerDecoder(embed_dim, num_heads, n_decoders, dim_conv, kernel, dropout, activation, norm)
        self.decoder_nums = n_decoders

        self.to_token = nn.Sequential(
            norm_layer(embed_dim),
            activation_layer,
            nn.Conv2d(embed_dim, num_embeds, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, c=None):
        x = self.token_embed(x).permute(0, 3, 1, 2)
        x_pos_embed_mask = torch.ones_like(x)[:, 0, :, :]
        x_pos = self.pos_embed(x, x_pos_embed_mask)
        x_pos = rearrange(x_pos, 'b c h w -> b (h w) c')
        outs = self.encoder_trans(x, pos=x_pos)
        out = outs[-1]
        c = c if c !=None else out
        if self.decoder_nums > 0:
            out = self.decoder_trans(c, out, pos=x_pos, query_pos=x_pos)
        out = self.to_token(out)

        return out


class Transformer(nn.Module):
    def __init__(self, input_nc, embed_dim=512, output_nc=512, dim_conv=2048, kernel=3, num_heads=8, n_encoders=12,
                 n_decoders=12, dropout=0., activation='gelu', norm='pixel', embed_type='learned'):
        super(Transformer, self).__init__()

        norm_layer = base_function.get_norm_layer(norm)
        activation_layer = base_function.get_nonlinearity_layer(activation)
        self.token_embed = base_function.PartialConv2d(input_nc, embed_dim, kernel_size=1, stride=1, padding=0, return_mask=True)
        self.pos_embed = build_position_embed(embed_type=embed_type, feats_dim=embed_dim)
        self.drop = nn.Dropout(dropout)
        self.encoder_trans = TransformerEncoder(embed_dim, num_heads, n_encoders, dim_conv, kernel, dropout, activation, norm)
        self.decoder_trans = TransformerDecoder(embed_dim, num_heads, n_decoders, dim_conv, kernel, dropout, activation, norm)
        self.decoder_nums = n_decoders

        self.to_token = nn.Sequential(
            norm_layer(embed_dim),
            activation_layer,
            nn.Conv2d(embed_dim, output_nc, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, mask=None, bool_mask=True):
        x, mask = self.token_embed(x, mask)
        x_pos_embed_mask = torch.ones_like(x)[:, 0, :, :]
        x_pos = self.pos_embed(x, x_pos_embed_mask)
        x_pos = rearrange(x_pos, 'b c h w -> b (h w) c')
        mask = torch.max(mask, 1e-2 * torch.ones_like(mask))
        key_padding_mask = rearrange(mask, 'b c h w -> b (c h w)')
        outs = self.encoder_trans(x, pos=x_pos, src_key_padding_mask=key_padding_mask, bool_mask=bool_mask)
        out = outs[-1]
        if self.decoder_nums > 0:
            out = self.decoder_trans(out, out, pos=x_pos, query_pos=x_pos)
        out = self.to_token(out)

        return out


######################################################################################
# base transformer structure
######################################################################################
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=6, dim_conv=2048, kernel=3, dropout=0.,
                 activation='gelu', norm='pixel'):
        super(TransformerEncoder, self).__init__()
        layer = TransformerEncoderLayer(embed_dim, num_heads, dim_conv, kernel, dropout, activation, norm)
        self.layers = _get_clones(layer, num_layers)

    def forward(self, src, src_key_padding_mask=None, src_mask=None, pos=None, bool_mask=True):
        out = src
        outs = []
        src_key_padding_mask_bool = src_key_padding_mask
        for i, layer in enumerate(self.layers):
            if src_key_padding_mask is not None:
                src_key_padding_mask_bool = src_key_padding_mask < 0.5 if bool_mask else src_key_padding_mask
                src_key_padding_mask = src_key_padding_mask ** 0.5
            out = layer(out, src_key_padding_mask_bool, src_mask, pos)
            outs.append(out)
        return outs


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=6, dim_conv=2048, kernel=3, dropout=0.,
                 activation='gelu', norm='pixel'):
        super(TransformerDecoder, self).__init__()
        layer = TransformerDecoderLayer(embed_dim, num_heads, dim_conv, kernel, dropout, activation, norm)
        self.layers = _get_clones(layer, num_layers)
        self.nums = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, pos=None, query_pos=None, bool_mask=True):
        out = tgt
        if not isinstance(memory_key_padding_mask, list):
            if memory_key_padding_mask is not None and bool_mask:
                memory_key_padding_mask_bool = [memory_key_padding_mask ** (0.5 ** i) < 0.2 for i in range(self.nums)]
            else:
                memory_key_padding_mask_bool = [memory_key_padding_mask for _ in range(self.nums)]
        for i, layer in enumerate(self.layers):
            memory_i = memory[self.nums - i - 1] if isinstance(memory, list) else memory
            out = layer(out, memory_i, tgt_mask, memory_mask, tgt_key_padding_mask,
                        memory_key_padding_mask_bool[self.nums-i-1], pos, query_pos)

        return out


######################################################################################
# base transformer operation
######################################################################################
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dim_conv=2048, kernel=3, dropout=0., activation='gelu', norm='pixel'):
        """
        Encoder transformer block
        :param embed_dim: total dimension of the model
        :param num_heads: parallel attention heads
        :param dim_conv: feature in feedforward layer
        :param kernel: kernel size for feedforward operation, kernel=1 is similar to MLP layer
        :param dropout: a dropout layer on attention weight
        :param activation: activation function
        :param norm: normalization layer
        """
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.conv1 = base_function.PartialConv2d(embed_dim, dim_conv, kernel_size=kernel, padding=int((kernel-1)/2))
        self.conv2 = base_function.PartialConv2d(dim_conv, embed_dim, kernel_size=1, padding=0)

        self.norm1 = base_function.get_norm_layer(norm)(embed_dim)
        self.norm2 = base_function.get_norm_layer(norm)(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = base_function.get_nonlinearity_layer(activation)

    def _with_pos_embed(self, x, pos=None):
        return x if pos is None else x + pos

    def forward(self, src, src_key_padding_mask=None, src_mask=None, pos=None):
        b, c, h, w = src.size()
        src2 = self.norm1(src)
        src2 = rearrange(src2, 'b c h w->b (h w) c')
        q = k = self._with_pos_embed(src2, pos)
        src2 = self.attn(q, k, src2, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        src2 = rearrange(src2, 'b (h w) c->b c h w', h=h, w=w)
        src = src + self.dropout(src2)
        src2 = self.norm2(src)
        src2 = self.conv2(self.dropout(self.activation(self.conv1(src2))))
        src = src + self.dropout(src2)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dim_conv=2048, kernel=3, dropout=0., activation='gelu', norm='pixel'):
        """
        decoder transform model
        :param embed_dim: total dimension of the model
        :param num_heads: parallel attention heads
        :param dim_conv: feature in feedforward layer
        :param kernel: kernel size for feedforward operation, kernel=1 is similar to MLP layer
        :param dropout: a dropout layer on attention weight
        :param activation: activation function
        :param norm: normalization layer
        """
        super(TransformerDecoderLayer, self).__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.cross = MultiheadAttention(embed_dim, num_heads, dropout)
        self.conv1 = base_function.PartialConv2d(embed_dim, dim_conv, kernel_size=kernel, padding=int((kernel - 1) / 2))
        self.conv2 = base_function.PartialConv2d(dim_conv, embed_dim, kernel_size=1, padding=0)

        self.norm1 = base_function.get_norm_layer(norm)(embed_dim)
        self.norm2 = base_function.get_norm_layer(norm)(embed_dim)
        self.norm3 = base_function.get_norm_layer(norm)(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = base_function.get_nonlinearity_layer(activation)

    def _with_pos_embed(self, x, pos=None):
        return x if pos is None else x + pos

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, pos=None, query_pos=None):
        b, c, h, w = tgt.size()
        tgt2 = self.norm1(tgt)
        tgt2 = rearrange(tgt2, 'b c h w -> b (h w) c')
        q = k = self._with_pos_embed(tgt2, query_pos)
        tgt2 = self.attn(q, k, tgt2, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)
        tgt2 = rearrange(tgt2, 'b (h w) c ->b c h w', h=h, w=w)
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = rearrange(tgt2, 'b c h w ->b (h w) c')
        memory = rearrange(memory, 'b c h w ->b (h w) c')
        tgt2 = self.cross(q=self._with_pos_embed(tgt2, query_pos), k=self._with_pos_embed(memory, pos),
                          v=memory, key_padding_mask=memory_key_padding_mask, attn_mask=memory_mask)
        tgt2 = rearrange(tgt2, 'b (h w) c -> b c h w', h=h, w=w)
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.conv2(self.dropout(self.activation(self.conv1(tgt2))))
        tgt = tgt + self.dropout(tgt2)

        return tgt


class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information from different position"""
    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.bias = bias
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        if self.bias:
            nn.init.constant_(self.to_q.bias, 0.)
            nn.init.constant_(self.to_k.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        b, n, c, h = *q.shape, self.num_heads
        # calculate similarity map
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q = rearrange(q, 'b n (h d)->b h n d', h=h)
        k = rearrange(k, 'b n (h d)->b h n d', h=h)
        v = rearrange(v, 'b n (h d)->b h n d', h=h)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # assign the attention weight based on the mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            if key_padding_mask.dtype == torch.bool:
                dots = dots.masked_fill(key_padding_mask, float('-inf'))
            else:
                dots = torch.where(dots > 0, key_padding_mask * dots, dots/(key_padding_mask+1e-5))
        # calculate the attention value
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        # projection
        out = torch.einsum('bhij, bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])