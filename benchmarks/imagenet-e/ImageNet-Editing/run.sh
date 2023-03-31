#!/bin/sh
# rm -rf results
# mkdir results
# rm -rf tmp
# mkdir tmp
ls /usr/local/cuda*

# Backgrounds
bg_scale=$1 # 
bg_detemined=$2 # given the input background
hard=False
if [ "$1" != "" ]; then
    if [ $1 > 0 ]; then
        hard=True
    fi
fi

# Size
size=$3

# Direction
angle=$4

# Steps
tot_steps=100
step=$5
skip_step=`expr $tot_steps - $step`

# number of generated image
num_of_Images=$6

# Background removal
cd object_removal/TFill/
python test.py \
--name imagenet \
--img_file ../../tmp/img/ \
--mask_file ../../tmp/mask/ \
--results_dir ../../results \
--model tc \
--coarse_or_refine refine \
--gpu_id 0 \
--no_shuffle \
--batch_size 1 \
--preprocess scale_shortside \
--mask_type 3 \
--load_size 512 \
--attn_G \
--add_noise

cd ../../
mv results/imagenet/test_latest/img_ref_out/input_0.png results/object_removal.png
rm -rf results/imagenet/

# Resize
python resize_obj.py --img_path tmp/img/input.JPEG  --mask_path tmp/mask/input.png --scale $size

if [ "$2" != "" ]; then
    bg_path=$bg_detemined
else
    bg_path="../results/object_removal.png"
fi

echo "Background path: " echo $bg_path
echo "Steps: " echo $step
echo "Object pixel rate: " echo $size
echo "Object angle: " echo $angle

# Generating
cd editing_diffusion
if [ $1 > 0 ]; then
    CUDA_VISIBLE_DEVICES=7 python main.py -p "test.JPEG" -i $bg_path -i2 "../results/img_rescaled.png" --mask "../results/mask_rescaled.png" --output_path "../tmp" --batch_size 1 --skip_timesteps $skip_step --invert_mask --clip_guidance_lambda 0 --classifier_scale 0. --y 0 --final_save_root "../results/" --rotate_obj --angle $angle --background_complex $bg_scale --hard --iterations_num $num_of_Images # --coarse_to_fine #--background_preservation_loss # --vid #--clip_guidance_lambda 0
else
    CUDA_VISIBLE_DEVICES=7 python main.py -p "test.JPEG" -i $bg_path -i2 "../results/img_rescaled.png" --mask "../results/mask_rescaled.png" --output_path "../tmp" --batch_size 1 --skip_timesteps $skip_step --invert_mask --clip_guidance_lambda 0 --classifier_scale 0. --y 0 --final_save_root "../results/" --rotate_obj --angle $angle --background_complex $bg_scale --iterations_num $num_of_Images # --coarse_to_fine #--background_preservation_loss # --vid #--clip_guidance_lambda 0
fi



