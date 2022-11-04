# EasyRobust

<div align="center">

[![license](https://img.shields.io/github/license/alibaba/easyrobust.svg)](https://github.com/alibaba/easyrobust/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/alibaba/easyrobust.svg)](https://github.com/alibaba/easyrobust/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/alibaba/easyrobust.svg)](https://GitHub.com/alibaba/easyrobust/pull/)
[![GitHub latest release](https://badgen.net/github/release/alibaba/easyrobust)](https://GitHub.com/alibaba/easyrobust/releases/)
</div>

## What's New

- **[Oct 2022]**: [Towards Understanding and Boosting Adversarial Transferability from a Distribution Perspective](https://arxiv.org/abs/2210.04213) was accepted into TIP 2022! Codes will be avaliable at [examples/attacks/dra](examples/attacks/dra)

- **[Sep 2022]**: [Boosting Out-of-distribution Detection with Typical Features](https://arxiv.org/abs/2210.04200) was accepted into NeurIPS 2022! Codes avaliable at [examples/ood_detection/BATS](examples/ood_detection/BATS)

- **[Sep 2022]**: [Enhance the Visual Representation via Discrete Adversarial Training](https://arxiv.org/abs/2209.07735) was accepted into NeurIPS 2022! Codes avaliable at [examples/imageclassification/imagenet/dat](examples/imageclassification/imagenet/dat)

- **[Sep 2022]**: Updating 5 methods for analysing your robust models under [tools/](tools).

- **[Sep 2022]**: Updating 13 reproducing examples of robust training methods under [examples/imageclassification/imagenet](examples/imageclassification/imagenet).

- **[Sep 2022]**: Releasing 16 Adversarial Training models, including a Swin-B which achieves SOTA adversairal robustness with 47.42% on AutoAttack!

- **[Sep 2022]**: EasyRobust v0.2.0 released.

## Our Research Project

- **[TIP 2022]** Towards Understanding and Boosting Adversarial Transferability from a Distribution Perspective [[Paper](https://arxiv.org/abs/2210.04213), [Code](examples/attacks/dra)]
- **[NeurIPS 2022]** Boosting Out-of-distribution Detection with Typical Features [[Paper](https://arxiv.org/abs/2210.04200), [Code](examples/ood_detection/BATS)]
- **[NeurIPS 2022]** Enhance the Visual Representation via Discrete Adversarial Training [[Paper](https://arxiv.org/abs/2209.07735), [Code](examples/imageclassification/imagenet/dat)]
- **[CVPR 2022]** Towards Robust Vision Transformer [[Paper](https://arxiv.org/abs/2105.07926), [Code](examples/imageclassification/imagenet/rvt)]


## Introduction

EasyRobust is an **Easy**-to-use library for state-of-the-art **Robust** Computer Vision Research with [PyTorch](https://pytorch.org). EasyRobust aims to accelerate research cycle in robust vision, by collecting comprehensive robust training techniques and benchmarking them with various robustness metrics. The key features includes:

- **Reproducible implementation of SOTA in Robust Image Classification**: Most existing SOTA in Robust Image Classification are implemented - [Adversarial Training](https://arxiv.org/abs/1706.06083), [AdvProp](https://arxiv.org/abs/1911.09665), [SIN](https://arxiv.org/abs/1811.12231), [AugMix](https://arxiv.org/abs/1912.02781), [DeepAugment](https://arxiv.org/abs/2006.16241), [DrViT](https://arxiv.org/abs/2111.10493), [RVT](https://arxiv.org/abs/2105.07926), [FAN](https://arxiv.org/abs/2204.12451), [APR](https://arxiv.org/abs/2108.08487), [HAT](https://arxiv.org/abs/2204.00993), [PRIME](https://arxiv.org/abs/2112.13547), [DAT](https://arxiv.org/abs/2209.07735) and so on.

- **Benchmark suite**: Variety of benchmarks tasks including [ImageNet-A](https://arxiv.org/abs/1907.07174), [ImageNet-R](https://arxiv.org/abs/2006.16241), [ImageNet-Sketch](https://arxiv.org/abs/1905.13549), [ImageNet-C](https://arxiv.org/abs/1903.12261), [ImageNetV2](https://arxiv.org/abs/1902.10811), [Stylized-ImageNet](https://arxiv.org/abs/1811.12231), [ObjectNet](https://objectnet.dev/). 

- **Scalability**: You can use EasyRobust to conduct 1-gpu training, multi-gpu training on single machine and large-scale multi-node training.

- **Model Zoo**: Open source more than 30 pretrained adversarially or non-adversarially robust models. 

- **Analytical tools**: Support analysis and visualization about a pretrained robust model, including [Attention Visualization](./tools), [Decision Boundary Visualization](./tools), [Convolution Kernel Visualization](./tools), [Shape vs. Texture Biases Analysis](./tools), etc. Using these tools can help us to explain how robust training improves the interpretability of the model. 

## Technical Articles
We have a series of technical articles on the functionalities of EasyRobust.
 - [顶刊TIP 2022！阿里提出：从分布视角出发理解和提升对抗样本的迁移性](https://mp.weixin.qq.com/s/qtTXn3B4OYiBaZgHZo9cGA)
 - [无惧对抗和扰动、增强泛化，阿里安全打造更鲁棒的ViT模型，论文入选CVPR 2022](https://mp.weixin.qq.com/s/J6gqA09MxLwmN_C40Sjf1Q)

## Installation
### Install by PIP

clone EasyRobust repository:

```bash
$ git clone https://github.com/alibaba/easyrobust.git
```

setup from the source:
```bash
$ cd easyrobust
$ pip install -e .
```

or install from PyPI (not available yet):
```bash
$ pip install easyrobust
```

download the ImageNet dataset and place into `/path/to/imagenet`. Specify `$ImageNetDataDir` as ImageNet path by:

```bash
$ export ImageNetDataDir=/path/to/imagenet
```

**[Optional]:** If you use EasyRobust to evaluate the model robustness, download the benchmark dataset by:
```bash
$ sh download_data.sh
```

**[Optional]:** If you use analysis tools in `tools/`, install extra requirements by:
```bash
$ pip install -r requirements/optional.txt
```


### Docker
We have provided a runnable environment in `docker/Dockerfile` for users who do not want to install by pip. To use it, please confirm that `docker` and `nvidia-docker` have installed. Then run the following command:
```bash
docker build -t alibaba/easyrobust:v1 -f docker/Dockerfile . 
```

## Getting Started
EasyRobust focuses on the basic usages of: **(1) Evaluate and benchmark the robustness of a pretrained models** and **(2) Train your own robust models or reproduce the results of previous SOTA methods**.

### 1. How to evaluate and benchmark the robustness of given models?
It only requires a few lines to evaluate the robustness of a model using EasyRobust. We give a minimalist example in [benchmarks/resnet50_example.py](./benchmarks/resnet50_example.py):

```python
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
```
You can do evaluation by simply running the command: `python benchmarks/resnet50_example.py`. After running is completed, your will get the following output:
```
Top1 Accuracy on the ImageNet-Val: 76.1%
Top1 Accuracy on the ImageNet-A: 0.0%
Top1 Accuracy on the ImageNet-R: 36.2%
Top1 Accuracy on the ImageNet-Sketch: 24.1%
Top1 Accuracy on the ImageNet-V2: 63.2%
Top1 Accuracy on the Stylized-ImageNet: 7.4%
Top1 accuracy 39.2%, mCE: 76.7 on the ImageNet-C
Top1 Accuracy on the AutoAttack: 0.0%
```

### 2. How to use EasyRobust to train my own robust models?
We implement most robust training methods in the folder `examples/imageclassification/imagenet/`. All of them are based on a basic training script: [examples/imageclassification/imagenet/base_training_script.py](./examples/imageclassification/imagenet/base_training_script.py). By comparing the difference, you can clearly see where and which hyperparameters of basic training are modified to create a robust training example. Below we present the tutorials of some classic methods:
- [Adversarial Training on ImageNet using 8 GPUs](./examples/imageclassification/imagenet/adversarial_training)
- [AugMix Training on ImageNet with 180 Epochs](./examples/imageclassification/imagenet/augmix)
- [AdvProp for Improving Non-adversarial Robustness and Accuracy](./examples/imageclassification/imagenet/advprop)
- [Using Stylized ImageNet as Extended Data for Training](./examples/imageclassification/imagenet/SIN)
- [Discrete Adversarial Training for ViTs](./examples/imageclassification/imagenet/dat)
- [Training Robust Vision Transformers (RVT) with 300 Epochs](./examples/imageclassification/imagenet/rvt)
- [Robust Finetuning of CLIP Models](./examples/imageclassification/imagenet/wiseft)

## Analytical Tools

see [tools/README.md](./tools)

## Model Zoo and Baselines


### Submit your models
We provide a tool `benchmarks/benchmark.py` to help users directly benchmark their models:
```
Usage: 
    python benchmarks/benchmark.py [OPTIONS...]

OPTIONS:
    --model [ARCH in timm]
    --data_dir [PATH of the bencmark datasets]
    --ckpt_path [URL or PATH of the model weights]
```
If you are willing to submit the model to our benchmarks, you can prepare a python script similar to `benchmarks/benchmark.py` and weights file `xxx.pth`, zip all the files. Then open an issue with the "Submit Model" template and provide a json storing submit information. Below is a submission template in adversarial robustness benchmark of image classification:

```markdown
## Submit Json Information

{"date": "19/06/2017", 
 "extra_data": "no", 
 "model": "<b>Adversarial Training</b>", 
 "institution": "MIT", 
 "paper_link": "https://arxiv.org/abs/1706.06083", 
 "code_link": "", 
 "architecture": "swin-b", 
 "training framework": "easyrobust (v1)", 
 "ImageNet-val": 75.05, 
 "autoattack": 47.42, 
 "files": "<a href=http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_swin_base_patch4_window7_224_ep4.pth >download</a>", 
 "advrob_imgcls_leaderboard": true, 
 "oodrob_imgcls_leaderboard": false, 
 "advrob_objdet_leaderboard": false, 
 "oodrob_objdet_leaderboard": false}
```
We will check the result and present your result into the benchmark if there is no problem. For submission template of other benchmarks, check [submit-model.md](.github/ISSUE_TEMPLATE/submit-model.md).

Below is the model zoo and benchmark of the EasyRobust. All the results are runned by [benchmarks/adv_robust_bench.sh](./benchmarks/adv_robust_bench.sh) and [benchmarks/non_adv_robust_bench.sh](./benchmarks/non_adv_robust_bench.sh).

### Adversarial Robust Benchmark (sorted by AutoAttack)

| Training Framework | Method | Model | ImageNet-Val | AutoAttack | Files |
| ---- | :----: | :----: | :----: | :----: | :----: |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [Swin-B](https://arxiv.org/abs/2103.14030) | 75.05% | 47.42% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_swin_base_patch4_window7_224_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [Swin-S](https://arxiv.org/abs/2103.14030) | 73.41% | 46.76% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_swin_small_patch4_window7_224_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [ViT-B/16](https://arxiv.org/abs/2010.11929) | 70.64% | 43.04% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vit_base_patch16_224_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [EfficientNet-B3](https://arxiv.org/abs/1905.11946) | 67.65% | 41.72% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_efficientnet_b3_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [ResNet101](](https://arxiv.org/abs/1512.03385)) | 69.51% | 41.04% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_resnet101_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [ViT-S/16](https://arxiv.org/abs/2010.11929) | 66.43% | 39.20% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vit_small_patch16_224_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [EfficientNet-B2](https://arxiv.org/abs/1905.11946) | 64.75% | 38.54% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_efficientnet_b2_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [ResNeSt50d](https://arxiv.org/abs/2004.08955) | 70.03% | 38.52% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_resnest50d_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [ViT-B/32](https://arxiv.org/abs/2010.11929) | 65.58% | 37.38% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vit_base_patch32_224_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [EfficientNet-B1](https://arxiv.org/abs/1905.11946) | 63.99% | 37.20% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_efficientnet_b1_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [SEResNet101](https://arxiv.org/abs/1709.01507) | 71.11% | 37.18% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_seresnet101_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [ResNeXt50_32x4d](https://arxiv.org/abs/1611.05431) | 67.39% | 36.42% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_resnext50_32x4d_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [EfficientNet-B0](https://arxiv.org/abs/1905.11946) | 61.83% | 35.06% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_efficientnet_b0_ep4.pth) |
| [robustness](https://github.com/MadryLab/robustness) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [ResNet50](https://arxiv.org/abs/1512.03385) | 64.02% | 34.96% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/robustbench/robustness_advtrain_resnet50_linf_eps4.0.pth) |
| **EasyRobust (Ours)** | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [ResNet50](https://arxiv.org/abs/1512.03385) | 65.1% | 34.9% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/adversarial_training/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/adversarial_training/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/adversarial_training/summary.csv) | 
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [SEResNet50](https://arxiv.org/abs/1709.01507) | 66.68% | 33.56% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_seresnet50_ep4.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [DenseNet121](https://arxiv.org/abs/1608.06993) | 60.90% | 29.78% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_densenet121_ep4.pth) |
| [Official](https://github.com/mahyarnajibi/FreeAdversarialTraining) | [Free AT](https://arxiv.org/abs/1904.12843) | [ResNet50](https://arxiv.org/abs/1512.03385) | 59.96% | 28.58% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/robustbench/free_adv_step4_eps4_repeat4.pth) |
| [Official](https://github.com/locuslab/fast_adversarial) | [FGSM AT](https://arxiv.org/abs/2001.03994) | [ResNet50](https://arxiv.org/abs/1512.03385) | 55.62% | 26.24% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/robustbench/fastadv_imagenet_model_weights_4px.pth) |
| EasyRobust (V1) | [Adversarial Training](https://arxiv.org/abs/1706.06083) | [VGG16](https://arxiv.org/abs/1409.1556) | 59.96% | 25.92% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vgg16_ep4.pth) |

### Non-Adversarial Robust Benchmark (sorted by ImageNet-C)
| Training Framework | Method | Model | Files | ImageNet-Val | V2 | C (mCE↓) | R | A | Sketch| Stylized | ObjectNet |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| **EasyRobust (Ours)** | [DAT](http://arxiv.org/abs/2209.07735) | [ViT-B/16](https://arxiv.org/abs/2010.11929) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/dat/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/dat/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/dat/summary.csv) | 81.38% | 69.99% | 45.59 | 49.64% | 24.61% | 36.46% | 24.84% | 20.12% |
| **EasyRobust (Ours)** | - | [RVT-S*](https://arxiv.org/abs/2105.07926) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/rvt/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/rvt/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/rvt/summary.csv) | 82.10% | 71.40% | 48.22 | 47.84% | 26.93% | 35.34% | 20.71% | 23.24% |
| [Official](https://github.com/vtddggg/Robust-Vision-Transformer) | - | [RVT-S*](https://arxiv.org/abs/2105.07926) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/official_models/rvt_small_plus.pth) | 81.82% | 71.05% | 49.42 | 47.33% | 26.53% | 34.22% | 20.48% | 23.11% |
| **EasyRobust (Ours)** | - | [DrViT-S](https://arxiv.org/abs/2111.10493) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/drvit/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/drvit/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/drvit/summary.csv) | 80.66% | 69.62% | 49.96 | 43.68% | 20.79% | 31.13% | 17.89% | 20.50% |
| - | - | [DrViT-S](https://arxiv.org/abs/2111.10493) | - | 77.03% | 64.49% | 56.89 | 39.02% | 11.85% | 28.78% | 14.22% | 26.49% |
| [Official](https://github.com/amodas/PRIME-augmentations) | [PRIME](https://arxiv.org/abs/2112.13547) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/official_models/prime.pth) | 76.91% | 65.42% | 57.49 | 42.20% | 2.21% | 29.82% | 13.94% | 16.59% |
| **EasyRobust (Ours)** | [PRIME](https://arxiv.org/abs/2112.13547) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/prime/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/prime/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/prime/summary.csv) | 76.64% | 64.37% | 57.62 | 41.95% | 2.07% | 29.63% | 13.56% | 16.28% |
| **EasyRobust (Ours)** | [DeepAugment](https://arxiv.org/abs/2006.16241) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/deepaugment/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/deepaugment/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/deepaugment/summary.csv) | 76.58% | 64.77% | 60.27 | 42.80% | 3.62% | 29.65% | 14.88% | 16.88% |
| [Official](https://github.com/hendrycks/imagenet-r) | [DeepAugment](https://arxiv.org/abs/2006.16241) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/official_models/deepaugment.pth) | 76.66% | 65.24% | 60.37 | 42.17% | 3.46% | 29.50% | 14.68% | 17.13% |
| **EasyRobust (Ours)** | [Augmix](https://arxiv.org/abs/1912.02781) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/augmix/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/augmix/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/augmix/summary.csv) | 77.81% | 65.60% | 64.14 | 43.34% | 4.04% | 29.81% | 12.33% | 17.21% |
| **EasyRobust (Ours)** | [APR](https://arxiv.org/abs/2108.08487) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/apr/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/apr/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/apr/summary.csv) | 76.28% | 64.78% | 64.89 | 42.17% | 4.18% | 28.90% | 13.03% | 16.78% |
| [Official](https://github.com/google-research/augmix) | [Augmix](https://arxiv.org/abs/1912.02781) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/official_models/augmix.pth) | 77.54% | 65.42% | 65.27 | 41.04% | 3.78% | 28.48% | 11.24% | 17.54% |
| [Official](https://github.com/iCGY96/APR) | [APR](https://arxiv.org/abs/2108.08487) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/official_models/apr_sp.pth) | 75.61% | 64.24% | 65.56 | 41.35% | 3.20% | 28.37% | 13.01% | 16.61% |
| [Official](https://github.com/LiYingwei/ShapeTextureDebiasedTraining) | [S&T Debiased](https://arxiv.org/abs/2010.05981) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/official_models/debiased.pth) | 76.91% | 65.04% | 67.55 | 40.81% | 3.50% | 28.41% | 17.40% | 17.38% |
| **EasyRobust (Ours)** | [SIN+IN](https://arxiv.org/abs/1811.12231) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/sin/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/sin/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/sin/summary.csv) | 75.46% | 63.50% | 67.73 | 42.34% | 2.47% | 31.39% | 59.37% | 16.17% |
| [Official](https://github.com/rgeirhos/texture-vs-shape) | [SIN+IN](https://arxiv.org/abs/1811.12231) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/official_models/sin_in.pth) | 74.59% | 62.43% | 69.32 | 41.45% | 1.95% | 29.69% | 57.38% | 15.93% |
| [Non-Official](https://github.com/tingxueronghua/pytorch-classification-advprop) | [AdvProp](https://arxiv.org/abs/1911.09665) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/official_models/advprop.pth) | 77.04% | 65.27% | 70.81 | 40.13% | 3.45% | 25.95% | 10.01% | 18.23% |
| **EasyRobust (Ours)** | [S&T Debiased](https://arxiv.org/abs/2010.05981) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/shape_texture_debiased_training/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/shape_texture_debiased_training/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/shape_texture_debiased_training/summary.csv) | 77.21% | 65.10% | 70.98 | 38.59% | 3.28% | 26.09% | 14.59% | 16.99% |
| **EasyRobust (Ours)** | [AdvProp](https://arxiv.org/abs/1911.09665) | [ResNet50](https://arxiv.org/abs/1512.03385) | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/advprop/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/advprop/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/advprop/summary.csv) | 76.64% | 64.35% | 77.64 | 37.43% | 2.83% | 24.71% | 7.33% | 16.82% |

## Credits
EasyRobust concretizes previous excellent works by many different authors. We'd like to thank, in particular, the following implementations which have helped us in our development:
- [timm](https://github.com/rwightman/pytorch-image-models) @rwightman and the training script.
- [robustness](https://github.com/MadryLab/robustness) @MadryLab and [autoattack](https://github.com/fra31/auto-attack) @fra31 for attack implementation. 
- [modelvshuman](https://github.com/bethgelab/model-vs-human) @bethgelab for model analysis.
- [AdaIN](https://github.com/naoto0804/pytorch-AdaIN) @naoto0804 for style trnsfer and [VQGAN](https://github.com/CompVis/taming-transformers) @CompVis for image discretization. 
- All the authors and implementations of the robustness research work we refer in this library.  

## Citing EasyRobust

We provide a BibTeX entry for users who apply EasyRobust to help their research: 

```BibTeX
@misc{mao2022easyrobust,
  author =       {Xiaofeng Mao and Yuefeng Chen and Xiaodan Li and Gege Qi and Ranjie Duan and Rong Zhang and Hui Xue},
  title =        {EasyRobust: A Comprehensive and Easy-to-use Toolkit for Robust Computer Vision},
  howpublished = {\url{https://github.com/alibaba/easyrobust}},
  year =         {2022}
}
```
