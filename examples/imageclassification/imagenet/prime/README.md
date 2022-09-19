# PRIME

paper: https://arxiv.org/abs/2112.13547

## Examples

ResNet50 training on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/prime/main.py \
--data_dir=$ImageNetDataDir \
--model=tv_resnet50 \
--lr=0.02 \
--pretrained \
--batch-size=64 \
--pin-mem \
--epochs=100 \
--output=output/PRIME \
--experiment=tmp
```

## Pretrained Models
| Model | ImageNet-Val | V2 | C (mCE↓) | R | A | Sketch| Stylized | ObjectNet | Files |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [ResNet50](https://arxiv.org/abs/1512.03385) | 76.64% | 64.37% | 57.62 | 41.95% | 2.07% | 29.63% | 13.56% | 16.28% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/prime/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/prime/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/prime/summary.csv) |