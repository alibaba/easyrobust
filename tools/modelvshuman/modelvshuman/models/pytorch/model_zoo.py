#!/usr/bin/env python3
import torch
from timm.models import create_model

from ..registry import register_model
from ..wrappers.pytorch import PytorchModel
from ..wrappers.pytorch import TimmModel

def model_pytorch(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__[model_name](pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)

@register_model("pytorch")
def model_timm(model_name, *args):
    print(f"Creating model: {model_name}")
    model = create_model(
        model_name,
        pretrained=True,
        num_classes=1000
    )
    return TimmModel(model, model_name, *args)
