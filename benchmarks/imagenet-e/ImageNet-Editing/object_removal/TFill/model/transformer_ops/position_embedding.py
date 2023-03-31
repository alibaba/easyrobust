import torch
import torch.nn as nn
import math


######################################################################################
# position embedding
######################################################################################
class PositionEmbeddingLearned(nn.Module):
    """
    This is a learned version of the position embedding
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(32, num_pos_feats)
        self.col_embed = nn.Embedding(32, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, mask):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i).unsqueeze(0).repeat(h, 1, 1)
        y_emb = self.row_embed(j).unsqueeze(1).repeat(1, w, 1)
        pos = (x_emb + y_emb).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class PositionEmbeddingSine(nn.Module):
    """
    This is a standard version of the position embedding, very similar to the one used by the
    "Attention is all you need" paper, generalized to work on examples
    """
    def __init__(self, feats_dim=512, temperature=10000, normalize=False, scale=None):
        """
        explicitly encode the position using the sinusoid:
        PE(pos,2i) = sin(pos/temperature^(2*i/d_model))
        PE(pos,2i+1) = cos(pos/temperature^(2*i/d_model))
        :param feats_dim: the dimension of features, each dimension of the positional embedding to a sinusoid
        :param temperature: wavelengths from a geometric progression from scale
        :param normalize: whether to normalize the position to (0,1)
        :param scale: scale for the position embedding
        """
        super(PositionEmbeddingSine, self).__init__()
        self.feats_dim = feats_dim
        self.T = temperature
        self.norm = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        x_embed = mask.cumsum(1, dtype=torch.float32)
        y_embed = mask.cumsum(2, dtype=torch.float32)
        if self.norm:
            eps = 1e-5
            x_embed = x_embed / (x_embed[:, -1:, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.feats_dim, dtype=torch.float32, device=x.device)
        dim_t = self.T ** (2*(dim_t//2)/self.feats_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x[:, :, :, 0::2], pos_x[:, :, :, 1::2] = pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()
        pos_y[:, :, :, 0::2], pos_y[:, :, :, 1::2] = pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()
        pos = (pos_x + pos_y).permute(0, 3, 1, 2) * 0.5
        return pos


def build_position_embed(embed_type='learned', feats_dim=512, temperature=10000):
    if embed_type == 'sine':
        pos_embed = PositionEmbeddingSine(feats_dim, temperature, normalize=True)
    elif embed_type == 'learned':
        pos_embed = PositionEmbeddingLearned(feats_dim)
    else:
        raise ValueError(f"nor supported {embed_type}")
    return pos_embed
