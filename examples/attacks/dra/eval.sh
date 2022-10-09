CUDA_VISIBLE_DEVICES=0 python validate.py --input_dir ./gen_images/resnet50_DRA_eps16_1 --arch inceptionresnetv2 --batch-size 80 --defense_methods None

### arch : vgg19_bn densenet121 densenet201 resnet152 senet154 inceptionresnetv2 inceptionv4 inceptionv3 

### defense_methods:'None,Augmix,SIN,SIN-IN,Linf-0.5,Linf-1.0,L2-0.05,L2-0.1,L2-0.5,L2-1.0'
### Augmix: https://drive.google.com/file/d/1z-1V3rdFiwqSECz7Wkmn4VJVefJGJGiF/view
### SIN: https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar
### SIN-IN: https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar
### Download the defense models and save these models to the folder "defense_models".

### Linf and L2 robust models: https://github.com/microsoft/robust-models-transfer
### Download the robust models and save these models to the folder "robust_models".



