# DeepAugment

paper: https://arxiv.org/abs/2006.16241

## Examples

ResNet50 training on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/deepaugment/main.py \
--data_dir=$ImageNetDataDir \
--model=tv_resnet50 \
--lr=0.1 \
--pretrained \
--batch-size=128 \
--pin-mem \
--sched=cosine \
--epochs=30 \
--output=output/deepaugment \
--experiment=tmp
```

## Pretrained Models
| Model | ImageNet-Val | V2 | C (mCE↓) | R | A | Sketch| Stylized | ObjectNet | Files |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [ResNet50](https://arxiv.org/abs/1512.03385) | 76.58% | 64.77% | 60.27 | 42.80% | 3.62% | 29.65% | 14.88% | 16.88% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/deepaugment/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/deepaugment/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/deepaugment/summary.csv) |