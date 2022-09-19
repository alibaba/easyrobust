# Robust Vision Transformer

paper: https://arxiv.org/abs/2105.07926

## Examples

RVT-S* training on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/rvt/main.py \
--data_dir=$ImageNetDataDir \
--model=rvt_small_plus \
--amp \
--epochs=300 \
--batch-size=128 \
--lr=0.001 \
--drop-path=0.1 \
--model-ema \
--model-ema-decay=0.99996 \
--opt=adamw \
--opt-eps=1e-8 \
--weight-decay=0.05 \
--sched=cosine \
--warmup-lr=1e-6 \
--warmup-epochs=5 \
--cooldown-epochs=10 \
--patience-epochs=10 \
--color-jitter=0.4 \
--aa=rand-m9-mstd0.5-inc1 \
--smoothing=0.1 \
--reprob=0.25 \
--mixup=0.8 \
--cutmix=1.0 \
--pin-mem \
--output=output/rvt \
--experiment=tmp
```

## Pretrained Models
| Model | ImageNet-Val | V2 | C (mCE↓) | R | A | Sketch| Stylized | ObjectNet | Files |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [RVT-S*](https://arxiv.org/abs/2105.07926) | 82.10% | 71.40% | 48.22 | 47.84% | 26.93% | 35.34% | 20.71% | 23.24% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/rvt/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/rvt/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/rvt/summary.csv) |