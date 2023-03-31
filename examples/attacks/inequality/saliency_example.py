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
from PIL import Image

from index import *
from utils import * 
from visualize import *

import visualization as viz
from vis_tools.integrated_gradients import *

torch.manual_seed(6)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sns.set_style("whitegrid")

parser = argparse.ArgumentParser(description='PyTorch Inequality Test')
parser.add_argument('--data_dir', default='/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/val',
                    help='directory of test data')
parser.add_argument('--model_path', default='/root/robust_model_weights/lf/resnet50_linf_eps8.0.ckpt', help = 'Load the model from the path')
parser.add_argument('--result_dir', default='./vis-results',
                    help='Path of adv trained model for saving checkpoint')

args = parser.parse_args()

if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
perturb_ratio_list = []
top1_conf_drop_list = []
natural_gini_list = []
adv_gini_list = []
confidence_drop_list = []
positive_contribution_list = []
Gini_index = Index()
def get_pred_index(model, test_input):
    out = model(test_input)
    _, index = torch.max(out, 1)
    return index
def get_pred_and_confidence(model, test_input, idx2label):
    out = model(test_input)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return idx2label[index[0]], percentage[index[0]].item(), index

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

def get_post_saliency_map(img_grad_np, gap  = 10):
    w,h,c = img_grad_np.shape
    target = int(w//gap)
    for i in range(target):
        for j in range(target):
            right_w = min(img_grad_np.shape[0], (i+1)*gap)
            right_h = min(img_grad_np.shape[1], (j+1)*gap)
            img_grad_np[i*gap:right_w, j*gap:right_h, :] = img_grad_np[i*gap:right_w, j*gap:right_h, :].sum()
    # img_grad_np =  (img_grad_np - np.min(img_grad_np))/(np.max(img_grad_np) - np.min(img_grad_np))  
    d = np.std(img_grad_np)
    img_grad_np = np.clip(img_grad_np, img_grad_np.mean()- d,img_grad_np.mean()+d )
    return img_grad_np

def draw_saliency_map(test_input, img_grad, model_name, img_name, post = True):
    img_grad_np = img_grad.cpu().detach().numpy()[0]
    org_img = np.moveaxis(img_denorm(test_input)[0].cpu().detach().numpy(), 0, -1)
    attr_t = np.transpose(img_grad.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    gini =  Gini_index.gini(abs(np.sort(img_grad_np.flatten())[::-1]))
    if post:
        average_attr_t = get_post_saliency_map(attr_t,gap  = 2)
    else:
        average_attr_t = get_average_saliency_map(attr_t,gap  = 2)

    plt_fig, plt_ax = viz.visualize_image_attr(average_attr_t, org_img, method="blended_heat_map",sign="positive",
                       cmap = "coolwarm", alpha_overlay = 0.7,outlier_perc=1)
    save_path = os.path.join(args.result_dir, model_name + str(gini) +  img_name)
    # plt_ax.set_title(model_name+str(gini),y = -0.1, fontweight="bold", fontsize = 14)
    plt_fig.savefig(save_path, bbox_inches = 'tight', pad_inches= 0.05)
    
    
def draw_normalized_importance_distribution(natural_img_gradshap, adv_lf_gradshap, adv_l2_gradshap, result_dir):
    plt.figure(figsize=(12,6))

    
    adv_l2_gs_np = np.sort(adv_l2_gradshap.cpu().detach().numpy().flatten())[::-1]
    # natural_gs_np = natural_gs_np[natural_gs_np>0][::-1]
    adv_l2_gini = Gini_index.gini(abs(adv_l2_gs_np))
    # adv_l2_gs_np = adv_l2_gs_np/np.max(abs(adv_l2_gs_np))
    # adv_gs_np = adv_gs_np[adv_gs_np>0]
    plt.fill_between( np.arange(len(adv_l2_gs_np)), 0, adv_l2_gs_np, color = "g", alpha = 0.5,  label="Adv. L2. trained")    
    
    
    natural_gs_np = np.sort(natural_img_gradshap.cpu().detach().numpy().flatten())[::-1]
    # natural_gs_np = natural_gs_np[natural_gs_np>0][::-1]
    natural_gini = Gini_index.gini(abs(natural_gs_np))
    # natural_gs_np = natural_gs_np/np.max(abs(natural_gs_np))
    # natural_gs_np = natural_gs_np[natural_gs_np>0]
    
    
#     plt.fill_between( np.arange(len(aug_gs_np)), 0, aug_gs_np, color = "g", alpha=0.5, label='Aug.') 
    plt.fill_between( np.arange(len(natural_gs_np)), 0, natural_gs_np, color = "b", alpha = 0.5,  label="Std. trained")
    
    adv_lf_gs_np = np.sort(adv_lf_gradshap.cpu().detach().numpy().flatten())[::-1]
    # natural_gs_np = natural_gs_np[natural_gs_np>0][::-1]
    adv_gini = Gini_index.gini(abs(adv_lf_gs_np))
    # adv_lf_gs_np = adv_lf_gs_np/np.max(abs(adv_lf_gs_np))
    # adv_gs_np = adv_gs_np[adv_gs_np>0]
    plt.fill_between( np.arange(len(adv_lf_gs_np)), 0, adv_lf_gs_np, color = "r", alpha = 0.5,  label="Adv. Lf. trained")
    
    plt.xticks([])
    plt.ylabel('Significance',fontweight='bold',fontsize = 12)
    # plt.xlabel('Features (RGB)')
    plt.title('Noramlized Features Attribution Distribution.', fontweight='bold', y = -0.2, fontsize = 14)
    
    plt.legend(fontsize = 12)

    # dir_path = os.path.join(result_dir, str(img_id))
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    plt.savefig(os.path.join(result_dir, "normalized_feature_gini.png"))
    plt.close()

if __name__ == "__main__":
    # Load dataset
    class_idx = json.load(open("./imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]

    print("Loading dataset...")
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_data = image_folder_custom_label(root=args.data_dir, transform=transform_test, idx2label=class2label)
    test_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=1, shuffle=True)


    print("Loading model...")

    ds = ImageNet('/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/')
    

    natural_model = models.resnet50(pretrained=True)
    natural_model = nn.DataParallel(natural_model).to(device)
    natural_model.eval()
    #validate clean accuracy
    adv_model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path = args.model_path )
    adv_model = adv_model.model.to(device)
    adv_model = nn.DataParallel(adv_model).to(device)
    adv_model.eval()


    target_cnt = 10

    cur_cnt = 0

    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        test_input, test_target = inputs.to(device), targets.to(device)

        # img_path = os.path.join(args.test_data_dir, img_name)
        # img = Image.open(img_path)
        # img_t = transform_test(img)
        # batch_t = torch.unsqueeze(img_t, 0).type(torch.cuda.FloatTensor).to(device)
        natural_pred = get_pred_and_confidence(natural_model, test_input, idx2label)
        adv_pred = get_pred_and_confidence(adv_model, test_input, idx2label)

        if natural_pred[2]== test_target and adv_pred[2]== test_target:   
            img_name = str(cur_cnt) + '.jpg'

            natural_img_gradshap = get_gradshap(natural_model, test_input, test_target)
            draw_saliency_map(test_input, natural_img_gradshap, "natural_model_", img_name, False)
             #### Post-processing, compared visualization in robustness may be at odds with accuracy.
            draw_saliency_map(test_input, natural_img_gradshap, "natural_model_post_", img_name, True)
            
            adv_img_gradshap = get_gradshap(adv_model, test_input, test_target)                           
            draw_saliency_map(test_input, adv_img_gradshap, "adv_model_", img_name, False)
              #### Post-processing, compared visualization in robustness may be at odds with accuracy.          
            draw_saliency_map(test_input, adv_img_gradshap, "adv_model_post", img_name, True)



            img = np.moveaxis(img_denorm(test_input)[0].cpu().detach().numpy(), 0, -1)*255.0
            img_path = os.path.join(args.result_dir, "org_img_{}.jpg".format(str(cur_cnt)))
            draw_org_img(test_input,img_path)


            cur_cnt+=1
            print(cur_cnt)
            if cur_cnt >= target_cnt:
                break
            
