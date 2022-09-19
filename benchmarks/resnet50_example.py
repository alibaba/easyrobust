import torch
import torchvision
from easyrobust.benchmarks import *

#############################################################
#         Define your model
#############################################################
model = torchvision.models.resnet50(pretrained=True)
model = model.eval()
if torch.cuda.is_available(): model = model.cuda()

#############################################################
#         Start Evaluation
#############################################################

# ood
evaluate_imagenet_val(model, 'benchmarks/data/imagenet-val')
evaluate_imagenet_a(model, 'benchmarks/data/imagenet-a')
evaluate_imagenet_r(model, 'benchmarks/data/imagenet-r')
evaluate_imagenet_sketch(model, 'benchmarks/data/imagenet-sketch')
evaluate_imagenet_v2(model, 'benchmarks/data/imagenetv2')
evaluate_stylized_imagenet(model, 'benchmarks/data/imagenet-style')
evaluate_imagenet_c(model, 'benchmarks/data/imagenet-c')
# objectnet is optional since it spends a lot of disk storage. we skip it here. 
# evaluate_objectnet(model, 'benchmarks/data/ObjectNet/images')

# adversarial
evaluate_imagenet_autoattack(model, 'benchmarks/data/imagenet-val')

