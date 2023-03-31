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
from resnet import resnet34

from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

from index import *
from utils import * 
from visualize import *
from prune import *

from visualize import *

import visualization as viz
from vis_tools.integrated_gradients import *
torch.manual_seed(1234)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# sns.set_style("whitegrid")

parser = argparse.ArgumentParser(description='PyTorch Inequality Test')
parser.add_argument('--data_dir', default='/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/val',
                    help='directory of test data')
parser.add_argument('--model_path', default='/root/robust_model_weights/lf/resnet50_linf_eps8.0.ckpt')
parser.add_argument('--result_dir', default='./occlusion_results',
                    help='Path of adv trained model for saving checkpoint')
args = parser.parse_args()

if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

perturb_ratio_list = []
top1_conf_drop_list = []
gini_list = []
confidence_drop_list = []

Gini_index = Index()

class OcclusionAttack():
    r"""    
    Arguments:
        model (nn.Module): model to attack.
        
    """
    def __init__(self, model, cnt =21, r_min =1, r_max = 21):
        self.steps = steps
        self.targeted = targeted
        self.batch_size = batch_size
        self.model = model
        self.r_min = r_min
        self.r_max = r_max
        self.alpha = alpha
        self.occlusion_cnt = cnt
        occ_prameters_list = []
        for r in range(self.r_min, self.r_max):
            for c in range(1, self.occlusion_cnt):
                occ_prameters_list.append((r,c))
        self.occ_prameters_list = occ_prameters_list
    
        
    def get_perturb_mask(self,img_grad_np, img,r, c):
        perturb_value_list = []
        perturb_center_list = []
        img_np = img.cpu().detach().numpy()[0]
        w, h = img_grad_np.shape
        perturb_mask = np.zeros((3, w, h))
        avg_value_mask = np.zeros((3, w, h))
        img_grad_flatten = np.sort(img_grad_np.flatten())
        # print(img_grad_flatten[:5])
        for i in range(c):
            perturb_value_list.append(img_grad_flatten[-1*(i+1)])
        for value in perturb_value_list:
            x,y = np.argwhere(img_grad_np == value)[0]
            perturb_mask[:, max(0, x-r):min(w, x+r), max(0, y-r):min(h, y+r)] = 1               
            # for _ in range(3):
            #     avg_value_mask[_, max(0, x-r):min(w, x+r), max(0, y-r):min(h, y+r)] =  np.mean(img_np[_, max(0, x-r):min(w, x+r), max(0, y-r):min(h, y+r)])
            avg_value_mask[:, max(0, x-r):min(w, x+r), max(0, y-r):min(h, y+r)] =  np.min(img_np)
        perturb_mask = torch.from_numpy(np.expand_dims(perturb_mask, axis = 0)).to(device)
        avg_value_mask = torch.from_numpy(np.expand_dims(avg_value_mask, axis = 0)).to(device)
        # perturb_mask = torch.from_numpy(perturb_mask).to(device)
        return perturb_mask, avg_value_mask
        
    def get_feature_map_max(self, img_grad):
        img_grad_np = np.max(img_grad.cpu().detach().numpy()[0], axis = 0)
        return img_grad_np
    
    def get_feature_map_avg(self, img_grad):
        img_grad_np = np.mean(img_grad.cpu().detach().numpy()[0], axis = 0)
        return img_grad_np

        
    def occlusion(self, images, labels, img_grad, occ_type = "black"):
        
        org_images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)
        
        img_grad_np = self.get_feature_map_avg(img_grad)
        occ_value = org_images.min()
        if occ_type == "black":
            occ_value = org_images.min()
        elif occ_type == "grey":
            n = 0
        for (r, c) in self.occ_prameters_list:
            perturb_mask, occ_value = self.get_perturb_mask(img_grad_np,images, r, c)
            occ_images = org_images.detach()*(1-perturb_mask) + perturb_mask*occ_value
            outputs = self.model(occ_images.type(torch.cuda.FloatTensor))
            pecent_all = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            percentage = pecent_all[int(labels)].item()
            _, pre = torch.max(outputs.data, 1)
            if self.targeted:
                suc_rate = ((pre == labels).sum()/self.batch_size).cpu().detach().numpy()
            else:
                suc_rate = ((pre != labels).sum()/self.batch_size).cpu().detach().numpy()

            if suc_rate >= 1:
                print('End at r: {}, c: {}, with suc. rate {}'.format(r,c, suc_rate))
                return occ_images.detach(), perturb_mask, r, c, pre

        return occ_images.detach(), perturb_mask, r, c, pre
        

def get_pred_and_confidence(model, test_input, idx2label):
    out = model(test_input)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return idx2label[index[0]], percentage[index[0]].item(), index

def draw_saliency_map(test_input, img_grad, model_name, img_name, post = True):
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
    img_grad_np = img_grad.cpu().detach().numpy()[0]
    org_img = np.moveaxis(img_denorm(test_input)[0].cpu().detach().numpy(), 0, -1)
    attr_t = np.transpose(img_grad.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    gini =  Gini_index.gini(abs(np.sort(img_grad_np.flatten())[::-1]))
    if post:
        average_attr_t = get_3d_average_saliency_map(attr_t,gap  = 2)
    else:
        average_attr_t = get_average_saliency_map(attr_t,gap  = 2)

    plt_fig, plt_ax = viz.visualize_image_attr(average_attr_t, org_img, method="blended_heat_map",sign="positive",
                       cmap = "coolwarm", alpha_overlay = 0.7,outlier_perc=1)
    save_path = os.path.join(args.result_dir, model_name + str(gini) +  img_name)
    # plt_ax.set_title(model_name+str(gini),y = -0.1, fontweight="bold", fontsize = 14)
    plt_fig.savefig(save_path, bbox_inches = 'tight', pad_inches= 0.05)
    

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
    ds = ImageNet('/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/')
    print("Loading model...")
    natural_model = models.resnet50(pretrained=True)
    natural_model = nn.DataParallel(natural_model).to(device)
    natural_model.eval()
    #validate clean accuracy
    adv_model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path =  '/root/robust_model_weights/lf/resnet50_linf_eps1.0.ckpt' )
    adv_model = adv_model.model.to(device)
    adv_model = nn.DataParallel(adv_model).to(device)
    adv_model.eval()

    adv_model_1, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path =  '/root/robust_model_weights/lf/resnet50_linf_eps2.0.ckpt' )
    adv_model_1 = adv_model_1.model.to(device)
    adv_model_1 = nn.DataParallel(adv_model_1).to(device)
    adv_model_1.eval()
    
    adv_model_2, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path =  '/root/robust_model_weights/lf/resnet50_linf_eps8.0.ckpt' )
    adv_model_2 = adv_model_2.model.to(device)
    adv_model_2 = nn.DataParallel(adv_model_2).to(device)
    adv_model_2.eval()
    
#     adv_model_3, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path = '/root/robust_model_weights/lf/resnet50_linf_eps1.0.ckpt' )
#     adv_model_3 = adv_model_3.model.to(device)
#     adv_model_3 = nn.DataParallel(adv_model_3).to(device)
#     adv_model_3.eval()
    
#     adv_model_4, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path = '/root/robust_model_weights/lf/resnet50_linf_eps2.0.ckpt' )
#     adv_model_4 = adv_model_4.model.to(device)
#     adv_model_4 = nn.DataParallel(adv_model_4).to(device)
#     adv_model_4.eval()
    
#     adv_model_5, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path = '/root/robust_model_weights/lf/resnet50_linf_eps8.0.ckpt' )
#     adv_model_5 = adv_model_5.model.to(device)
#     adv_model_5 = nn.DataParallel(adv_model_5).to(device)
#     adv_model_5.eval()


    #validate clean accuracy
    
#     model_name = args.model_path.split('/')[-1].split('.')[0]
    color_mode = "black"
    result_dir = os.path.join(args.result_dir, color_mode)

    
    target_cnt = 1000
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    cur_cnt = 0
    natural_suc_cnt = 0
    adv_suc_cnt = 0
    adv_1_suc_cnt = 0
    adv_2_suc_cnt = 0
    
    search_step = 0
    occlusion_cnt = 11
    
    
    natural_occlusion_pixels = 0
    adv_occlusion_pixels = 0
    natural_occlusion_attack = OcclusionAttack(natural_model, cnt = occlusion_cnt)
    adv_occlusion_attack = OcclusionAttack(adv_model, cnt = occlusion_cnt)
    adv_occlusion_attack_1 = OcclusionAttack(adv_model_1, cnt = occlusion_cnt) 
    adv_occlusion_attack_2 = OcclusionAttack(adv_model_2, cnt = occlusion_cnt) 
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        test_input, test_target = inputs.to(device), targets.to(device)

        natural_pred = get_pred_and_confidence(natural_model, test_input, idx2label)
        adv_pred = get_pred_and_confidence(adv_model, test_input, idx2label)
        adv_pred_1 = get_pred_and_confidence(adv_model_1, test_input, idx2label)
        adv_pred_2 = get_pred_and_confidence(adv_model_2, test_input, idx2label)


        if natural_pred[2]== test_target and adv_pred[2]== test_target and  adv_pred_1[2]== test_target and  adv_pred_2[2]== test_target:   
            
            cur_cnt += 1
            print("Current cnt:",cur_cnt)
            
            natural_img_gradshap = get_gradshap(natural_model, test_input, test_target)                                   
            natural_occ_imgs, perturb_mask, r, c, pre = natural_occlusion_attack.occlusion(test_input, test_target, natural_img_gradshap, color_mode)           
            natural_suc_cnt += int(pre!=test_target)
            
   

            adv_img_gradshap = get_gradshap(adv_model, test_input, test_target)
                           
            adv_occ_imgs, perturb_mask, r, c, pre = adv_occlusion_attack.occlusion(test_input, test_target, adv_img_gradshap, color_mode)
            adv_suc_cnt += int(pre!=test_target)
   
            
            adv_img_gradshap_1 = get_gradshap(adv_model_1, test_input, test_target)                     
            adv_occ_imgs, perturb_mask, r, c, pre = adv_occlusion_attack_1.occlusion(test_input, test_target, adv_img_gradshap_1, color_mode)
            adv_1_suc_cnt += int(pre!=test_target)           

            adv_img_gradshap_2 = get_gradshap(adv_model_2, test_input, test_target)                     
            adv_occ_imgs, perturb_mask, r, c, pre = adv_occlusion_attack_2.occlusion(test_input, test_target, adv_img_gradshap_2, color_mode)
            adv_2_suc_cnt += int(pre!=test_target)                       
                
            print( "Natural suc rate: ", natural_suc_cnt/cur_cnt, "Adv suc rate: ",adv_suc_cnt/cur_cnt, "Adv 1 suc rate: ", adv_1_suc_cnt/cur_cnt, "Adv 2 suc rate: ",adv_2_suc_cnt/cur_cnt) 

            
#             if cur_cnt <= 10:
            
#                 natural_pred_name, natural_pred_confidence,_ = get_pred_and_confidence(natural_model, natural_occ_imgs.type(torch.cuda.FloatTensor), idx2label)

#                 natural_title_name = "Pred.: "+ natural_pred_name +", Conf.: " + str( "{:.2f}".format(natural_pred_confidence))     
#                 natural_img_path = os.path.join(args.result_dir, "natural_occ_img_{}_{}_{}_{}.png".format(cur_cnt, r, c, last_name))
#                 draw_occ_img(natural_occ_imgs, natural_title_name, natural_img_path)
#                 draw_saliency_map(test_input, natural_img_gradshap, "natural_model_", img_name, False)   

#                 org_img_path = os.path.join(args.result_dir, "org_img_{}.png".format(cur_cnt))
#                 draw_occ_img(test_input, " ", org_img_path)

                
#             mask_path = os.path.join(result_dir, "mask_{}_{}_{}_{}.png".format(cur_cnt, r, c, title_name))

#             feature_map_3d_path = os.path.join(result_dir, "feature_{}_{}_{}_{}.png".format(cur_cnt, r, c,title_name))
            # draw_org_img(adv_imgs, img_path)
#             draw_mask(perturb_mask, mask_path)
#             draw_3d_importance_distribution(inputs, natural_img_gradshap, cur_cnt, feature_map_3d_path)
            # if pre != test_target:
            #     suc_cnt += 1
            #     occlusion_pixels += perturb_mask[0].sum().item()
            # print("Average perturb num: ", occlusion_pixels/suc_cnt)
            # print("Current suc rate: ",suc_cnt/cur_cnt )
            # draw_org_img(adv_imgs, cur_cnt, img_result_dir)
            # draw_org_img(perturb_mask, cur_cnt, mask_result_dir)
            
        if cur_cnt == target_cnt:
            break
    print("---------- IG----------------")
    print("Black: Natural Model,", "Occlusion_pixel_num_avg: ", natural_occlusion_pixels/target_cnt, "Suc. rate: ", natural_suc_cnt/cur_cnt )
    print("Black: Natural Model resnet50_linf_eps8.0,", "Occlusion_pixel_num_avg: ", adv_occlusion_pixels/target_cnt, "Suc. rate: ", adv_suc_cnt/cur_cnt )