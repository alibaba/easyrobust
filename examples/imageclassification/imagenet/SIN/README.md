# Stylized-ImageNet Training

paper: https://arxiv.org/abs/1811.12231

## Examples

ResNet50 training on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/SIN/main.py \
--data_dir=$ImageNetDataDir \
--model=tv_resnet50 \
--pretrained \
--batch-size=32 \
--pin-mem \
--lr=0.1 \
--epochs=90 \
--output=output/SIN \
--experiment=tmp
```

## Pretrained Models
| Model | ImageNet-Val | V2 | C (mCE↓) | R | A | Sketch| Stylized | ObjectNet | Files |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [ResNet50](https://arxiv.org/abs/1512.03385) | 75.46% | 63.50% | 67.73 | 42.34% | 2.47% | 31.39% | 59.37% | 16.17% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/sin/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/sin/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/sin/summary.csv) |