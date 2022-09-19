# HAT

paper: https://arxiv.org/abs/2204.00993

## Examples

ViT-S training on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/hat/main.py \
--data_dir=$ImageNetDataDir \
--model=vit_small_patch16_224 \
--apex-amp \
--epochs=300 \
--batch-size=128 \
--lr=0.0016 \
--opt=adamw \
--weight-decay=0.05 \
--sched=cosine \
--warmup-lr=1e-6 \
--warmup-epochs=20 \
--cooldown-epochs=10 \
--patience-epochs=10 \
--color-jitter=0.4 \
--aa=rand-m9-mstd0.5-inc1 \
--smoothing=0.1 \
--reprob=0.25 \
--mixup=0.8 \
--cutmix=1.0 \
--pin-mem \
--output=output/hat \
--experiment=tmp
```

## Pretrained Models
todo
