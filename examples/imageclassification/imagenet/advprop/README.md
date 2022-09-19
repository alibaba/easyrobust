# Adversarial Examples Improve Image Recognition

paper: https://arxiv.org/abs/1911.09665

## Examples

ResNet50 training on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/advprop/main.py \
--data_dir=$ImageNetDataDir \
--lr=0.1 \
--opt=momentum \
--batch-size=32 \
--pin-mem \
--epochs=105 \
--dist-bn='' \
--output=output/advprop \
--experiment=tmp
```

## Pretrained Models
| Training Framework | Method | Model | ImageNet-Val | V2 | C (mCE↓) | R | A | Sketch| Stylized | ObjectNet | Files |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| **EasyRobust (Ours)** | [AdvProp](https://arxiv.org/abs/1911.09665) | [ResNet50](https://arxiv.org/abs/1512.03385) | 76.64% | 64.35% | 77.64 | 37.43% | 2.83% | 24.71% | 7.33% | 16.82% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/advprop/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/advprop/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/advprop/summary.csv) |