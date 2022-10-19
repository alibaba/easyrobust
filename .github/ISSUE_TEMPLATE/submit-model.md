---
name: Submit Model
about: To submit a robust model in benchmark
title: "[Submit Model] <Your_Method_Name>"
labels: submit-model
assignees: ''

---

## Submit Json Information

{}

## Example Submit Json format is below:

### Adversarial Robustness of Image Classification
```
{"date": "19/06/2017", 
 "extra_data": "no", 
 "model": "<b>Adversarial Training</b>", 
 "institution": "MIT", 
 "paper_link": "https://arxiv.org/abs/1706.06083", 
 "code_link": "", 
 "architecture": "swin-b", 
 "training framework": "easyrobust (v1)", 
 "ImageNet-val": 75.05, 
 "autoattack": 47.42, 
 "files": "<a href=http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_swin_base_patch4_window7_224_ep4.pth >download</a>", 
 "advrob_imgcls_leaderboard": true, 
 "oodrob_imgcls_leaderboard": false, 
 "advrob_objdet_leaderboard": false, 
 "oodrob_objdet_leaderboard": false}
```

### OOD Robustness of Image Classification

```
{"date": "27/12/2021", 
 "extra_data": "no", 
 "institution": "EPFL", 
 "paper_link": "https://arxiv.org/abs/2112.13547", 
 "model": "<b>PRIME</b>", 
 "code_link": "", 
 "architecture": "resnet50", 
 "training framework": "official", 
 "ImageNet-val": 76.91, 
 "imagenet-v2": 65.42, 
 "imagenet-c": 57.49, 
 "imagenet-r": 42.2, 
 "imagenet-a": 2.21, 
 "imagenet-sketch": 29.82, "stylized-imagenet": 13.94, 
 "objectnet": 16.59, 
 "files": "<a href=http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/official_models/prime.pth >download</a>", 
 "advrob_imgcls_leaderboard": false,
 "oodrob_imgcls_leaderboard": true, 
 "advrob_objdet_leaderboard": false, 
 "oodrob_objdet_leaderboard": false}
```

### Adversarial Robustness of Object Detection
```
{"date": "19/06/2017", 
 "extra_data": "no", 
 "model": "<b>MTD</b>", 
 "institution": "", 
 "paper_link": "", 
 "code_link": "", 
 "architecture": "ssd", 
 "training framework": "official", "coco_test": 24.2, 
 "pgd_cls": 13.0, 
 "pgd_loc": 13.4, 
 "CWA": 7.7, 
 "files": "", 
 "advrob_imgcls_leaderboard": false, 
 "oodrob_imgcls_leaderboard": false, 
 "advrob_objdet_leaderboard": true, 
 "oodrob_objdet_leaderboard": false}
```

### OOD Robustness of Object Detection
```
{"date": "19/06/2017", 
 "extra_data": "no", 
 "model": "<b>-</b>", 
 "institution": "", 
 "paper_link": "", 
 "code_link": "", 
 "architecture": "Faster R-CNN (RNXT-101-32x4d-FPN-DCN)", 
 "training framework": "mmdetection", 
 "coco_test": 43.4, 
 "coco_c": 26.7, 
 "rpc": 61.6, 
 "files": "", 
 "advrob_imgcls_leaderboard": false, 
 "oodrob_imgcls_leaderboard": false, 
 "advrob_objdet_leaderboard": false, 
 "oodrob_objdet_leaderboard": true}
```