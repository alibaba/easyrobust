# Adversarial Training

paper: https://arxiv.org/abs/1706.06083

## Examples

ResNet50 training on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/adversarial_training/main.py \
--data_dir=$ImageNetDataDir \
--smoothing=0.0 \
--pin-mem \
--lr=0.2 \
--output=output/adversarial_training \
--experiment=tmp
```

## Pretrained Models
| Model | ImageNet-Val | AutoAttack | Files |
| ---- | :----: | :----: | :----: |
| [ResNet50](https://arxiv.org/abs/1512.03385) | 65.1% | 34.9% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/adversarial_training/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/adversarial_training/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/adversarial_training/summary.csv) | 