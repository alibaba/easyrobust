# EasyRobust
An Easy-to-use Framework for Large-scale Robust Training

## News

- March, 2022: RVT was accepted in CVPR 2022.

## Release
### Toolkits
> [**EasyRobust**](https://github.com/alibaba/easyrobust/tree/main/easyrobust): a toolkit for training your robust models, with a collection of **adversarially robust** / **non-adversarially robust** / **standard** training method on **CNN** / **ViT** architectures.

### Image Classification
> [**RVT**](https://github.com/alibaba/easyrobust/tree/main/RVT): strong baseline for robustness research on vision transformers. Paper link: [Towards Robust Vision Transformer](https://arxiv.org/abs/2105.07926)

### Objection Detection
> [**D^2ETR**](https://github.com/alibaba/easyrobust/tree/main/ddetr): decoder-only DETR with computationally efficient cross-scale attention on transformer backbone. Paper link: [D^2ETR: Decoder-Only DETR with Computationally Efficient Cross-Scale Attention](https://arxiv.org/abs/2203.00860) 

------

## A Collection of ImageNet Robust Models

EasyRobust also contains the robust models trained on ImageNet, and the scripts for robustness evaluation.

The benchmarked results have been contained in ARES-Bench. 

### Usage

First, clone the repository locally:
```
git clone https://github.com/alibaba/easyrobust.git
cd easyrobust
pip install -r requirements.txt
```
Then test runing on ImageNet Validation set:
```
python robustness_validation.py --model=resnet50 --interpolation=3 --imagenet_val_path=/path/to/ILSVRC/Data/CLS-LOC/val
```
The trained models will be downloaded automaticly. If you want to download the checkpoints manually, check the urls in [utils.py](https://github.com/alibaba/easyrobust/blob/main/easyrobust/test_scripts/utils.py).

The code supports [Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet), [ImageNet-V2](https://github.com/modestyachts/ImageNetV2), [ImageNet-R](https://github.com/hendrycks/imagenet-r), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), [ObjectNet](https://objectnet.dev/), [ImageNet-C](https://github.com/hendrycks/robustness), [AutoAttack](https://github.com/fra31/auto-attack) evaluation. See [test_example.sh](https://github.com/alibaba/easyrobust/blob/main/easyrobust/test_scripts/test_example.sh) for details. 

### Adversarially robust models
18 Adversarially trained models are opened in `utils.py`. 
| Architecture   | $\epsilon$ | Clean Accuracy | AutoAttack Robust Accuracy  | Weights |
|:-------:|:--------:|:--------:|:--------:|:--------:|
| vgg13 |  4/255 | 55.44 | 23.08 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vgg13_ep4.pth) |
| vgg16 |  4/255 | 59.96 | 25.92 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vgg16_ep4.pth) |
| densenet121 |  4/255 | 60.90 | 29.78 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_densenet121_ep4.pth) |
| seresnet50 |  4/255 | 66.68 | 33.56 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_seresnet50_ep4.pth) |
| resnet50 |  4/255 | 66.07 | 34.08 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_resnet50_ep4.pth) |
| efficientnet_b0 |  4/255 | 61.83 | 35.06 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_efficientnet_b0_ep4.pth) |
| resnext50_32x4d |  4/255 | 67.39 | 36.42 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_resnext50_32x4d_ep4.pth) |
| seresnet101 |  4/255 | 71.11 | 37.18 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_seresnet101_ep4.pth) |
| efficientnet_b1 |  4/255 | 63.99 | 37.20 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_efficientnet_b1_ep4.pth) |
| vit_base_patch32_224 |  4/255 | 65.58 | 37.38 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vit_base_patch32_224_ep4.pth) |
| resnest50d |  4/255 | 70.03 | 38.52 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_resnest50d_ep4.pth) |
| efficientnet_b2 |  4/255 | 64.75 | 38.54 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_efficientnet_b2_ep4.pth) |
| vit_small_patch16_224 |  4/255 | 66.43 | 39.20 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vit_small_patch16_224_ep4.pth) |
| resnet101 |  4/255 | 69.51 | 41.04 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_resnet101_ep4.pth) |
| efficientnet_b3 |  4/255 | 67.65 | 41.72 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_efficientnet_b3_ep4.pth) |
| vit_base_patch16_224 |  4/255 | 70.64 | 43.04 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vit_base_patch16_224_ep4.pth) |
| swin_small_patch4_window7_224 |  4/255 | 73.41 | 46.76 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_swin_small_patch4_window7_224_ep4.pth) |
| swin_base_patch4_window7_224 |  4/255 | 75.05 | 47.42 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_swin_base_patch4_window7_224_ep4.pth) |

### Non-Adversarially robust models

We collect some non-adversarially robust models based on resnet50. To test these models, replace the [this line](https://github.com/alibaba/easyrobust/blob/db87c8f26a2b722ba5af1de4e6b9aebba76de6de/utils.py#L5) with following urls:

| Method   |  Architecture | Clean Accuracy | Common Corruption (mCE)↓  | Weights |
|:-------:|:--------:|:--------:|:--------:|:--------:|
| `SIN` |  resnet50 | 74.59 | 69.32 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/SIN.pth) |
| `DebiasedCNN` | resnet50 | 76.91 | 67.55 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/res50-debiased.pth) |
| `Augmix` |  resnet50 | 77.54 | 65.27 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/augmix.pth) |
| `ANT` |  resnet50 | 76.07 | 63.37 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/ANT3x3_Model.pth) |
| `DeepAugment` |  resnet50 | 76.66 | 60.37 | [url](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/deepaugment.pth) |
