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
torch.manual_seed(8)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# sns.set_style("whitegrid")

parser = argparse.ArgumentParser(description='PyTorch Inequality Test')
parser.add_argument('--data_dir', default='/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/val',
                    help='directory of test data')
parser.add_argument('--model_path', default='/root/robust_model_weights/lf/resnet50_linf_eps8.0.ckpt')
parser.add_argument('--result_dir', default='./noise_results',
                    help='Path of adv trained model for saving checkpoint')
args = parser.parse_args()

if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

perturb_ratio_list = []
top1_conf_drop_list = []
gini_list = []
confidence_drop_list = []


def get_pred_and_confidence(model, test_input, idx2label):
    out = model(test_input)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return idx2label[index[0]], percentage[index[0]].item()



def inequality_sparse_attack(model, img_grad, images, labels,  pixel_threhold = 10, random_noise = True):
    
    adv_images = images.clone().detach().to(device)
    org_images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    # batch_size = list(adv_images.shape)[0]
    img_grad_np = img_grad.cpu().detach().numpy()

    batch_size, c,w, h = img_grad_np.shape
#     perturb_mask = np.zeros((batch_size, c,w, h))
#     img_grad_np_channel = np.max(img_grad_np, axis = 1)

    perturb_value = np.sort(img_grad_np.flatten())[::-1][pixel_threhold]
    # print( np.sort(img_grad_np.flatten())[:3])
    # print("Test pixel num: ", pixel_threhold, "Perturb value: ", perturb_value)
    perturb_mask =torch.from_numpy((img_grad_np>=perturb_value) * 1.0).to(device)
    # perturb_mask = torch.from_numpy(perturb_mask).to(device)

    # print("Test image range: ",images.min(), images.max())


    # Starting at a uniformly random point
    if random_noise:
        adv_images = adv_images + perturb_mask * torch.empty_like(adv_images).uniform_(images.min().item(), images.max().item())
    else:
        adv_images = adv_images*(1-perturb_mask) + perturb_mask * torch.empty_like(adv_images).uniform_(images.min().item(), images.max().item())
    adv_images = torch.clamp(adv_images, min=images.min().item(), max=images.max().item()).detach()
    outputs = model(adv_images.type(torch.cuda.FloatTensor))
    pecent_all = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    percentage = pecent_all[int(labels)].item()
    _, pre = torch.max(outputs.data, 1)
    return adv_images.detach(), perturb_mask, 0, pre
    

def save_noise_result(noise_img, img, cur_cnt, perturb_pixel_num, save_type = "natural"):
    noise = torch.clamp(noise_img - img,  min=img.min().item(), max=img.max().item())
    if save_type == "natural":
        img_name = "natural_noise_{}_{}.png".format(cur_cnt, perturb_pixel_num)
    else:
        img_name = "adv_noise_{}_{}.png".format(cur_cnt, perturb_pixel_num)
    title_name = ''
    img_path = os.path.join(args.result_dir, img_name)
    draw_occ_img(noise, title_name, img_path)
def save_result(model, img, idx2label, cur_cnt, perturb_pixel_num, suc_name, save_type = "natural"):
    pred_name, pred_confidence = get_pred_and_confidence(model, img.type(torch.cuda.FloatTensor), idx2label)
    title_name = "Pred.: "+ pred_name +", Conf.: " + str( "{:.2f}".format(pred_confidence))
    if save_type == "natural":
        img_name = "natural_{}_{}_{}.png".format(cur_cnt, perturb_pixel_num, suc_name)
    else:
        img_name = "adv_{}_{}_{}.png".format(cur_cnt, perturb_pixel_num, suc_name)
    img_path = os.path.join(args.result_dir, img_name)
    draw_occ_img(img, title_name, img_path)
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
    natural_model = models.resnet50(pretrained=True).to(device)
    natural_model = nn.DataParallel(natural_model).to(device)
    natural_model.eval()
    

    ds = ImageNet('/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/')
    
    # natural_model, _ = make_and_restore_model(arch='wide_resnet50_2', dataset=ds,resume_path = '/root/robust_model_weights/lf/wide_resnet50_2_linf_eps4.0.ckpt' )
    # natural_model = natural_model.model
    # natural_model = natural_model.to(device)
    # natural_model.eval()
    
    # adv_l2_model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path = '/root/robust_model_weights/l2/resnet50_l2_eps5.ckpt' )
    # adv_l2_model = adv_l2_model.model
    # adv_l2_model = adv_l2_model.to(device)
    # adv_l2_model.eval()
    
    adv_model, _ = make_and_restore_model(arch='resnet50', dataset=ds,resume_path = args.model_path )
    adv_model = adv_model.model.to(device)
    # adv_model = nn.DataParallel(adv_model).to(device)
    adv_model.eval()


    #validate clean accuracy
    
    model_name = args.model_path.split('/')[-1].split('.')[0]
    # model_name = "resnet18_clean"
    result_dir = os.path.join(args.result_dir, model_name)
    
    Gini_index = Index()
    target_cnt = 1000
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    cur_cnt = 0
    search_step = 0
    natural_suc_cnt = 0
    adv_suc_cnt = 0
    
    # perturb_pixel_list = [1000, 5000]
    perturb_pixel_num = 5000
    perturb_tot_num = 0
    natural_gini_list = []
    adv_gini_list = []
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        test_input, test_target = inputs.to(device), targets.to(device)
        natural_pred = get_pred(natural_model.type(torch.cuda.FloatTensor), test_input, test_target)
        adv_pred = get_pred(adv_model.type(torch.cuda.FloatTensor), test_input, test_target)

        if natural_pred[1] and  adv_pred[1]:
            cur_cnt += 1
            print("Current cnt:",cur_cnt)

            
            
            natural_img_gradshap = get_gradshap(natural_model, test_input, test_target) 

            adv_imgs_natural, perturb_mask, cur_search, pre = inequality_sparse_attack(natural_model,natural_img_gradshap, test_input, test_target,  pixel_threhold = perturb_pixel_num)

            natural_suc_cnt += (pre != test_target).sum().cpu().detach().numpy()

            natural_gs_np_positive = natural_img_gradshap.cpu().detach().numpy()
            
            natural_gs_np = np.sort(natural_gs_np_positive.flatten())[::-1]

            last_name = ""
            if pre!=test_target:            
                last_name = "success"
            else:
                last_name = "fail"
            # save_result(natural_model, adv_imgs_natural, idx2label, cur_cnt, perturb_pixel_num, last_name, save_type = "natural")
            # save_noise_result(adv_imgs_natural, test_input, cur_cnt, perturb_pixel_num, save_type = "natural")

            adv_img_gradshap = get_gradshap(adv_model, test_input, test_target)

            adv_imgs_adv, perturb_mask, cur_search, pre = inequality_sparse_attack(adv_model,adv_img_gradshap, test_input, test_target,  pixel_threhold = perturb_pixel_num)
            # pre, correct =  get_pred(natural_model.type(torch.cuda.FloatTensor), adv_imgs_adv.type(torch.cuda.FloatTensor), test_target)
            adv_suc_cnt += (pre != test_target).sum().cpu().detach().numpy()

#             adv_gs_np_positive = adv_img_gradshap.cpu().detach().numpy()
            
#             adv_gs_np = np.sort(adv_gs_np_positive.flatten())[::-1]

            if pre!=test_target:            
                last_name = "success"
            else:
                last_name = "fail" 
            # save_result(adv_model, adv_imgs_adv, idx2label, cur_cnt, perturb_pixel_num,  last_name, save_type = "adv")
            # save_noise_result(adv_imgs_adv, test_input, cur_cnt, perturb_pixel_num, save_type = "natural")
            print("Natural suc rate: ", natural_suc_cnt/cur_cnt, "Adv suc rate: ", adv_suc_cnt/cur_cnt)

            search_step += cur_search
            # suc_cnt += (pre != test_target).sum().cpu().detach().numpy()
            # print("Current natural success rate: ", natural_suc_cnt/cur_cnt, "Adv: ", adv_suc_cnt/cur_cnt)
            # print("Aversage perturb num: ", perturb_tot_num/suc_cnt)
            # draw_org_img(adv_imgs, cur_cnt, img_result_dir)
            # draw_org_img(perturb_mask, cur_cnt, mask_result_dir)
            
        if cur_cnt == target_cnt:
            break
    print("Attack Pixel threhold: ", perturb_pixel_num)
    print("Natural gini: ", np.mean(natural_gini_list), np.std(natural_gini_list))
    print("Adv gini: ", np.mean(adv_gini_list), np.std(adv_gini_list))