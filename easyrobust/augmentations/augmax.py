"""
@inproceedings{wang2021augmax,
  title={AugMax: Adversarial Composition of Random Augmentations for Robust Training},
  author={Wang, Haotao and Xiao, Chaowei and Kossaifi, Jean and Yu, Zhiding and Anandkumar, Anima and Wang, Zhangyang},
  booktitle={NeurIPS},
  year={2021}
}
"""

import torch
import torch.nn as nn

class AugMaxModule(nn.Module):
    def __init__(self, device='cuda'):
        super(AugMaxModule, self).__init__()
        self.device = device

    def forward(self, xs, m, q):
        '''
        Args:
            xs: tuple of Tensors. len(x)=3. xs = (x_ori, x_aug1, x_aug2, x_aug3). x_ori.size()=(N,W,H,C)
            m: Tensor. m.size=(N)
            q: Tensor. q.size()=(N,3). w = softmax(q)
        '''
        
        x_ori = xs[0]
        w = torch.nn.functional.softmax(q, dim=1) # w.size()=(N,3)

        N = x_ori.size()[0]

        x_mix = torch.zeros_like(x_ori).to(self.device)
        for i, x_aug in enumerate(xs[1:]):
            wi = w[:,i].view((N,1,1,1)).expand_as(x_aug)
            x_mix += wi * x_aug 

        m = m.view((N,1,1,1)).expand_as(x_ori)
        x_mix = (1-m) * x_ori + m * x_mix

        return x_mix