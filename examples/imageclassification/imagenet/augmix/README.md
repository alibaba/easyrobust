# AugMix

paper: https://arxiv.org/abs/1912.02781

## Examples

ResNet50 training on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/augmix/main.py \
--data_dir=$ImageNetDataDir \
--batch-size=32 \
--pin-mem \
--lr=0.1 \
--epochs=180 \
--sched=cosine \
--output=output/augmix \
--experiment=tmp
```

## Pretrained Models
| Model | ImageNet-Val | V2 | C (mCE↓) | R | A | Sketch| Stylized | ObjectNet | Files |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [ResNet50](https://arxiv.org/abs/1512.03385) | 77.81% | 65.60% | 64.14 | 43.34% | 4.04% | 29.81% | 12.33% | 17.21% |[ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/augmix/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/augmix/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/augmix/summary.csv) |