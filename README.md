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

### Non-Adversarially robust models

We collect some non-adversarially robust models based on resnet50. To test these models, replace the [this line](https://github.com/alibaba/easyrobust/blob/db87c8f26a2b722ba5af1de4e6b9aebba76de6de/utils.py#L5) with following urls:

| Method   |  Architecture  | weights |
|:-------:|:--------:|:--------:|
| `SIN` |  resnet50 | http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/SIN.pth |
| `ANT` |  resnet50 | http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/ANT3x3_Model.pth |
| `Augmix` |  resnet50 | http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/augmix.pth |
| `DeepAugment` |  resnet50 | http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/deepaugment.pth |
| `DebiasedCNN` |  resnet50 | http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/res50-debiased.pth |
