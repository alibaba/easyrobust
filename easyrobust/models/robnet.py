"""
@article{guo2019meets,
  title={When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks},
  author={Guo, Minghao and Yang, Yuzhe and Xu, Rui and Liu, Ziwei and Lin, Dahua},
  journal={arXiv preprint arXiv:1911.10695},
  year={2019}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

code_robnet_large_v1 = code_robnet_large_v2 = [['11', '11',
                                      '11', '11',
                                      '11', '01',
                                      '11', '01',
                                      '10', '11',
                                      '01', '01',
                                      '01', '11']]

code_robnet_free = [['11', '10',
                '11', '11',
                '10', '11',
                '01', '01',
                '10', '11',
                '11', '00',
                '11', '01'],
               ['11', '00',
                '01', '10',
                '01', '00',
                '10', '10',
                '00', '01',
                '11', '10',
                '11', '00'],
               ['11', '11',
                '11', '01',
                '01', '11',
                '01', '00',
                '10', '00',
                '10', '10',
                '00', '11'],
               ['10', '11',
                '10', '01',
                '01', '10',
                '10', '01',
                '10', '11',
                '00', '00',
                '01', '10'],
               ['11', '11',
                '01', '00',
                '10', '10',
                '10', '01',
                '10', '01',
                '00', '10',
                '01', '11'],
               ['11', '10',
                '11', '11',
                '11', '01',
                '10', '11',
                '00', '10',
                '01', '11',
                '01', '11'],
               ['11', '11',
                '11', '10',
                '10', '01',
                '11', '10',
                '01', '10',
                '10', '10',
                '01', '10'],
               ['01', '11',
                '11', '11',
                '01', '11',
                '11', '11',
                '01', '11',
                '01', '11',
                '10', '00'],
               ['11', '11',
                '11', '11',
                '11', '01',
                '01', '11',
                '10', '01',
                '00', '10',
                '01', '11'],
               ['10', '11',
                '01', '00',
                '11', '11',
                '10', '11',
                '01', '11',
                '11', '11',
                '11', '00'],
               ['11', '10',
                '11', '00',
                '00', '00',
                '11', '00',
                '01', '10',
                '00', '01',
                '10', '11'],
               ['01', '11',
                '01', '11',
                '11', '10',
                '10', '11',
                '10', '11',
                '01', '11',
                '10', '00'],
               ['11', '11',
                '10', '10',
                '01', '00',
                '10', '11',
                '11', '01',
                '10', '10',
                '00', '01'],
               ['01', '11',
                '11', '11',
                '01', '01',
                '11', '01',
                '01', '11',
                '11', '01',
                '10', '10'],
               ['01', '11',
                '11', '11',
                '11', '01',
                '00', '01',
                '10', '10',
                '11', '10',
                '10', '11'],
               ['11', '11',
                '00', '11',
                '11', '01',
                '00', '01',
                '10', '00',
                '11', '01',
                '11', '11'],
               ['11', '11',
                '01', '11',
                '11', '10',
                '11', '10',
                '10', '10',
                '10', '10',
                '11', '11'],
               ['10', '11',
                '01', '11',
                '11', '01',
                '11', '00',
                '11', '11',
                '00', '10',
                '00', '01'],
               ['11', '10',
                '11', '11',
                '11', '11',
                '11', '10',
                '11', '00',
                '11', '01',
                '11', '11'],
               ['11', '11',
                '01', '01',
                '11', '11',
                '01', '00',
                '00', '10',
                '00', '01',
                '01', '11']]

operation_canditates = {
    '00': lambda C, stride: Zero(stride),
    '01': lambda C, stride: SepConv(C, C, 3, stride, 1),
    '10': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    '11': lambda C, stride: ResSepConv(C, C, 3, stride, 1),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=False)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=False),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=False)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ResSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ResSepConv, self).__init__()
        self.conv = SepConv(C_in, C_out, kernel_size, stride, padding)
        self.res = Identity() if stride == 1 else FactorizedReduce(C_in, C_out)

    def forward(self, x):
        return sum([self.conv(x), self.res(x)])

class ChosenOperation(nn.Module):

    def __init__(self, C, stride, genotype):
        super(ChosenOperation, self).__init__()
        self.op = operation_canditates[genotype](C, stride)

    def forward(self, x):
        return self.op(x)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, genotype):
        super(Cell, self).__init__()
        self.reduction = reduction
        self._steps = steps
        self._multiplier = multiplier

        # For search stage, the affine of BN should be set to False, in order to avoid conflict with architecture params
        self.affine = False

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self._ops = nn.ModuleList()
        self._complie(C, reduction, genotype)

    def _complie(self, C, reduction, genotype):
        offset = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = ChosenOperation(C, stride, genotype[offset + j])
                self._ops.append(op)
            offset += 2 + i

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, genotype_list, C=36, num_classes=10, layers=20, steps=4, multiplier=4, stem_multiplier=3,
                 share=False, AdPoolSize=1, ImgNetBB=False):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._share = share
        self._ImgNetBB = ImgNetBB

        if self._ImgNetBB:
            self.stem0 = nn.Sequential(nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(C // 2),
                                       nn.ReLU(inplace=False),
                                       nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(C))
            self.stem1 = nn.Sequential(nn.ReLU(inplace=False),
                                       nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(C))
            C_prev_prev, C_prev, C_curr = C, C, C
            reduction_prev = True

        else:
            C_curr = stem_multiplier * C
            self.stem = nn.Sequential(
                nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
            C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
            reduction_prev = False

        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                        genotype=genotype_list[0] if self._share else genotype_list[i])
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        if self._ImgNetBB:
            self.global_pooling = nn.AvgPool2d(7)
        else:
            self.global_pooling = nn.AdaptiveAvgPool2d(AdPoolSize)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        if self._ImgNetBB:
            s0 = self.stem0(input)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

def robnet_free(genotype_list=code_robnet_free, **kwargs):
    return Network(genotype_list=genotype_list, **kwargs)

def robnet_large_v1(genotype_list=code_robnet_large_v1, **kwargs):
    return Network(genotype_list=genotype_list, **kwargs)
