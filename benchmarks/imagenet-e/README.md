# ImageNet-E

The code and dataset will be released here.

- **[CVPR 2023]** ImageNet-E: Benchmarking Neural Network Robustness via Attribute Editing [[Paper](https://arxiv.org/abs/2303.17096), [Image editing toolkit](ImageNet-Editing), [ImageNet-E dataset](https://drive.google.com/file/d/19M1FQB8c_Mir6ermRsukTQReI-IFXeT0/view?usp=sharing)]



## Image Editing toolkit
To use the provided image editing toolkit, download the [checkpoints](https://drive.google.com/file/d/1qwDKS5HK8PRo5-baU-UCqsBXmRS2Ejes/view?usp=share_link) of diffusion models and [checkpoints](https://drive.google.com/file/d/16GuXCG4W9z614NLmOrlRnCawmJBkphOZ/view?usp=share_link) of TFill and unzip them in the corresponding path.
```bash
$ cd ImageNet-Editing
$ pip install -r requirements.txt
$ python app.py
```