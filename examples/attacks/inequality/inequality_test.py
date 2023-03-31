import os

import numpy as np

import torch
import torchvision
import torch.nn as nn
import cv2
import argparse
import seaborn as sns
import json
import torchvision
import torchvision.transforms as transforms
from torchvision import models


from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model


from index import *
from utils import * 
from visualize import *
from prune import *
torch.manual_seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sns.set_style("whitegrid")

parser = argparse.ArgumentParser(description='PyTorch Inequality Test')
parser.add_argument('--data_dir', default='/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/val',
                    help='directory of test data')
parser.add_argument('--model_path', default='/root/robust_model_weights/lf/resnet50_linf_eps8.0.ckpt', help = 'Load the model from the path')
parser.add_argument('--target_cnt', default=1000, help = 'Count of samples to evaluate')
parser.add_argument('--result_dir', default='./vis_results',
                    help='Path of adv trained model for saving checkpoint')
args = parser.parse_args()

Gini_index = Index()
if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
def get_pred_index(model, test_input):
    out = model(test_input)
    _, index = torch.max(out, 1)
    return index

def get_average_saliency_map(img_grad_np, gap  = 10):
    w,h,c = img_grad_np.shape
    target = int(w//gap)
    for i in range(target):
        for j in range(target):
            right_w = min(img_grad_np.shape[0], (i+1)*gap)
            right_h = min(img_grad_np.shape[1], (j+1)*gap)
            img_grad_np[i*gap:right_w, j*gap:right_h, :] = img_grad_np[i*gap:right_w, j*gap:right_h, :].sum()
    # img_grad_np =  (img_grad_np - np.min(img_grad_np))/(np.max(img_grad_np) - np.min(img_grad_np))     
    return img_grad_np

if __name__ == "__main__":
    # Load dataset
    class_idx = json.load(open("./imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]

    print("Loading dataset...")
    ds = ImageNet('/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/')
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_data = image_folder_custom_label(root=args.data_dir, transform=transform_test, idx2label=class2label)
    test_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=1, shuffle=True)

    print("Loading model...")
    natural_model = models.resnet50(pretrained=True)
    natural_model = nn.DataParallel(natural_model).to(device)
    natural_model.eval()

    adv_model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path = args.model_path)
    adv_model = adv_model.model.to(device)
    adv_model = nn.DataParallel(adv_model).to(device)
    adv_model.eval()
    


    cur_cnt = 0

    natural_gini_list = []
    adv_gini_list = []
    regional_natural_gini_list = []
    regional_adv_gini_list = []
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        test_input, test_target = inputs.to(device), targets.to(device)
        natural_pred = get_pred(natural_model, test_input, test_target)
        adv_pred = get_pred(adv_model, test_input, test_target)
        # Test correct samples
        if (natural_pred[1]==True) and (adv_pred[1]==True) :
#             

            natural_img_gradshap = get_gradshap(natural_model, test_input, test_target)
            natural_gs_np = natural_img_gradshap.cpu().detach().numpy()[0]
            natural_gini =  Gini_index.gini(abs(np.sort(natural_gs_np.flatten())[::-1]))
            natural_gini_list.append(natural_gini)

            natural_average_attr_t = get_average_saliency_map(natural_gs_np,gap  = 16)
            natural_average_attr_t = (natural_average_attr_t > 0) * natural_average_attr_t
            natural_gs_np_flatten = np.sort(natural_average_attr_t.flatten())[::-1]

            r_natural_gini = Gini_index.gini(natural_gs_np_flatten)
            regional_natural_gini_list.append(r_natural_gini)


            adv_img_gradshap = get_gradshap(adv_model, test_input, test_target)
            adv_gs_np = adv_img_gradshap.cpu().detach().numpy()[0]
            adv_gini =  Gini_index.gini(abs(np.sort(adv_gs_np.flatten())[::-1]))
            adv_gini_list.append(adv_gini)

            # adv_average_attr_t = (adv_average_attr_t > 0) * adv_average_attr_t
            adv_average_attr_t = (adv_gs_np>0)*adv_gs_np
            adv_average_attr_t = get_average_saliency_map(adv_average_attr_t,gap  = 8)
            adv_gs_np_flatten = np.sort(adv_average_attr_t.flatten())[::-1]

            r_adv_gini = Gini_index.gini(adv_gs_np_flatten)
            regional_adv_gini_list.append(r_adv_gini)
            
            
            if cur_cnt%50 ==0:
                print(cur_cnt)
                print("*********Std. Trained***********","Global Gini: ", np.mean(natural_gini_list), "Regional Gini: ", np.mean(regional_natural_gini_list))

                print("*********Adv. Trained***********","Global Gini: ", np.mean(adv_gini_list), "Regional Gini: ", np.mean(regional_adv_gini_list))

            cur_cnt += 1
            if cur_cnt == args.target_cnt:
                break
