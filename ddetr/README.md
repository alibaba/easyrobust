# D^2​ETR: Decoder-Only DETR

## Introduction

The PyTorch implementation of the paper *D^2ETR: Decoder-Only DETR with Computationally Efficient Cross-Scale Attention*

## Results

| Model             | Epoch | GFLOPs | Params | AP   |
| ----------------- | ----- | ------ | ------ | ---- |
| D^2​ETR            | 50    | 82     | 35     | 43.2 |
| Deformable D^2​ETR | 50    | 93     | 40     | 50.0 |

## Example usage

**Reuirements and Instation**

* [PyTorch](https://pytorch.org/) versiin >= 1.7.1
* [PVTv2-B2-Linear](https://github.com/whai362/PVT) pre-trained model
* Install other libraries

```
pip install -r requirements.txt
```

**Training**

* Train D^2​ETR on 8 GPUs

```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/ddetr.sh \
--coco_path /path/to/coco \
--pvt_resume /path/to/pvt
```

* Train Deformable D^2​ETR on 8 GPUs

```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/def_ddetr.sh \
--coco_path /path/to/coco \
--pvt_resume /path/to/pvt
```

**Evaluation**

* Evaluate D^2​ETR

```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/ddetr.sh \
--coco_path /path/to/coco \
--resume /path/to/model \
--eval
```

* Evaluate Deformable D^2​ETR

```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/def_ddetr.sh \
--coco_path /path/to/coco \
--resume /path/to/model \
--eval
```

## Citation

```
@misc{lin2022d2etr,
      title={D^2ETR: Decoder-Only DETR with Computationally Efficient Cross-Scale Attention}, 
      author={Junyu Lin and Xiaofeng Mao and Yuefeng Chen and Lei Xu and Yuan He and Hui Xue},
      year={2022},
      eprint={2203.00860},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```