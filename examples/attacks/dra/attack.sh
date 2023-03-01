#Generate adversarial examples with our DRA, PGD and SGM.

CUDA_VISIBLE_DEVICES=0 python attack.py --gamma 1 --output_dir ./gen_images/resnet50_DRA_eps16_1 --arch resnet50 --batch-size 80 --DRA 1

#CUDA_VISIBLE_DEVICES=0 python attack.py --gamma 1 --output_dir ./gen_images/resnet50_PGD_eps16_1 --arch resnet50 --batch-size 80 --gamma 1.0 --advertorch 1

#CUDA_VISIBLE_DEVICES=0 python attack.py --gamma 0.25 --output_dir ./gen_images/resnet50_SGM_eps16_1 --arch resnet50 --batch-size 80 --gamma 1.0 --advertorch 1


#Set --gamma 1.0 --advertorch 1 to use the baseline PGD attack.
#Set --gamma 0.25(for resnet),0.5(for densenet) --advertorch 1 to use the SOTA method SGM .(suggestion from the paper https://arxiv.org/abs/2002.05990)
#Set --DRA 1 to use DRA attack.
