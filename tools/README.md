# Usage of Analytical Tools

## Attention Visualization of CNNs

```
Usage: 
    python cnn_attention.py [OPTIONS...]

OPTIONS:
    --model [ARCH in timm]
    --ckpt_path [URL or PATH of the model weights]
    --input_image [PATH or URL of the input image]
    --method [METHOD of CNN visualization]
```

Examples:

```bash
python cnn_attention.py --model resnet50 --ckpt_path https://download.pytorch.org/models/resnet50-19c8e357.pth --input_image http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/assets/test.png 
```

then check `tools/cnn_attn.jpg`.


## Attention Visualization of ViTs

```
Usage: 
    python vit_attenton.py [OPTIONS...]

OPTIONS:
    --model [ARCH in timm]
    --ckpt_path [URL or PATH of the model weights]
    --input_image [PATH or URL of the input image]
```

Examples:

```bash
python vit_attenton.py --model vit_base_patch16_224 --ckpt_path http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/timm_model/vit_base_patch16_224.pth --input_image http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/assets/test.png 
```

then check `tools/vit_attn.png`.

## Convolution Kernel Visualization

```
Usage: 
    python kernal_visualization.py [OPTIONS...]

OPTIONS:
    --model [ARCH in timm]
    --ckpt_path [URL or PATH of the model weights]
```

Examples:

```bash
python kernal_visualization.py --model resnet50 --ckpt_path https://download.pytorch.org/models/resnet50-19c8e357.pth
```

then check `tools/vis_filters.png`.

## Shape vs. Texture Biases Analysis

Examples: 

```bash
python shape_texture_bias.py
```
then check `tools/modelvshuman/figures/example-figures/`.
