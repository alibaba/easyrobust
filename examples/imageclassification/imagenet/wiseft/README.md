# Robust fine-tuning of zero-shot models

paper: https://arxiv.org/abs/2109.01903

## Examples

Robust fine-tuning of CLIP ViT-B/16 model on ImageNet with single 8-GPU machine:

## Examples

Robust fintuning of ViT-B/16 on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/wiseft/main.py \
--data_dir=$ImageNetDataDir \
--model=clip_vit_base_patch16_224 \
--epochs=10 \
--workers=8 \
--batch-size=64 \
--lr=0.00003 \
--weight-decay=0.1 \
--opt=adamw \
--opt-eps=1e-8 \
--sched=cosine \
--clip-grad=1.0 \
--pin-mem \
--output=output/wiseft \
--experiment=tmp
```

## Pretrained Models
| Model1 | Model2 |
| :----: | :----: |
| [zeroshot.pt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/wiseft/zero_shot.pt) | [9.pt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/wiseft/9.pt) |