# Shape-Texture Debiased Neural Network Training

paper: https://arxiv.org/abs/2010.05981

## Examples

ResNet50 training on ImageNet with single 8-GPU machine:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/imageclassification/imagenet/shape_texture_debiased_training/main.py \
--data_dir=$ImageNetDataDir \
--lr=0.2 \
--batch-size=256 \
--warmup-epochs=5 \
--warmup-lr=0.0 \
--pin-mem \
--epochs=100 \
--output=output/shape_texture_debiased_training \
--experiment=tmp
```

## Pretrained Models
| Model | ImageNet-Val | V2 | C (mCE↓) | R | A | Sketch| Stylized | ObjectNet | Files |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [ResNet50](https://arxiv.org/abs/1512.03385) | 77.21% | 65.10% | 70.98 | 38.59% | 3.28% | 26.09% | 14.59% | 16.99% | [ckpt](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/shape_texture_debiased_training/model_best.pth.tar)/[args](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/shape_texture_debiased_training/args.yaml)/[logs](http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/shape_texture_debiased_training/summary.csv) |