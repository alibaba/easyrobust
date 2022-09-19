# Fully Attentional Networks

paper: https://arxiv.org/abs/2204.12451

## Examples

FAN-S training on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/fan/main.py \
--data_dir=$ImageNetDataDir \
--model=fan_small_12_p16_224 \
--amp \
--epochs=300 \
--batch-size=128 \
--lr=0.002 \
--drop-path=0.2 \
--model-ema \
--model-ema-decay=0.99992 \
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
--output=output/fan \
--experiment=tmp
```

## Pretrained Models
todo