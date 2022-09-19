import torch
import torch.nn as nn
from easyrobust.third_party.adain import StyleTransfer


class CrossStyleAugmentor(nn.Module):
    def __init__(self, device):
        super(CrossStyleAugmentor,self).__init__()

        self.device = device
        self.style_transfer = StyleTransfer(self.device)

    def forward(self, image, label, alpha, label_mix_alpha=0):
        assert (image >= 0.0).all() and (image <= 1.0).all(), 'input of CrossStyle Augmentation must be ranged from 0.0 to 1.0!'
        n, c, h, w = image.shape

        content = image.detach()
        random_index = torch.randperm(n)
        style = image.detach()[random_index]
        label_style = label.detach()[random_index]
        with torch.no_grad():
            stylized_image = self.style_transfer(content, style, alpha)

        return stylized_image, (label, label_style, torch.ones(n).to(self.device) * label_mix_alpha)
