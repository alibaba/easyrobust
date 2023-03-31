import numpy as np
import matplotlib.pyplot as plt
import os
from utils import * 
from index import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import seaborn as sns
# sns.set_style("darkgrid")

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
Gini_index = Index()

def img_denorm(x, mean= [0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def draw_perturb_loop(natural_perturb_tracer, img_id, result_dir):   
    plt.figure(figsize=(6,4))
    tot_len = len(natural_perturb_tracer["confidence"][0])
    if tot_len == (natural_perturb_tracer["correct_id"][0]):
        plt.plot(np.arange(0,natural_perturb_tracer["correct_id"][0]), natural_perturb_tracer["confidence"][0], c = "b")
    else:
        correct_y = natural_perturb_tracer["confidence"][0][:natural_perturb_tracer["correct_id"][0]+1]
        wrong_y = natural_perturb_tracer["confidence"][0][natural_perturb_tracer["correct_id"][0]:]
        label_name = "Natural, " + "Std-" + str( "{:.4f}".format(natural_perturb_tracer["std"][0]))

        plt.plot(np.arange(0,natural_perturb_tracer["correct_id"][0]+1), correct_y, c = "b")
        plt.plot(np.arange(natural_perturb_tracer["correct_id"][0], tot_len), wrong_y, c = "r", linestyle = "-.")

        plt.text(natural_perturb_tracer["correct_id"][0]-1, wrong_y[0]-1, 'X', horizontalalignment = "left", c="r", fontsize = 'x-large', fontweight = 'extra bold')
    
    plt.ylabel("Confidence.")
    plt.xlabel("Percentage of dropped pixels (‰).")
    plt.ylim(0,100)
    # plt.title("Confidence on target class.", fontweight='bold')
    # plt.legend()
    
    dir_path = os.path.join(result_dir, str(img_id))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    plt.savefig(os.path.join(dir_path, "pixel_drop_check.png"))
    plt.close()
    
    
    
def draw_normalized_importance_distribution(natural_img_gradshap, adv_img_gradshap, result_dir):
    plt.figure(figsize=(12,3))
    
    natural_gs_np = np.sort(natural_img_gradshap.cpu().detach().numpy().flatten())[::-1]
    # natural_gs_np = natural_gs_np[natural_gs_np>0][::-1]
    natural_gini = Gini_index.gini(abs(natural_gs_np))
    natural_gs_np = natural_gs_np/np.max(natural_gs_np)
    natural_gs_np = natural_gs_np[natural_gs_np>0]
    
    
#     plt.fill_between( np.arange(len(aug_gs_np)), 0, aug_gs_np, color = "g", alpha=0.5, label='Aug.') 
    plt.fill_between( np.arange(len(natural_gs_np)), 0, natural_gs_np, color = "b", alpha = 0.7,  label="Std. trained")
    
    adv_gs_np = np.sort(adv_img_gradshap.cpu().detach().numpy().flatten())[::-1]
    # natural_gs_np = natural_gs_np[natural_gs_np>0][::-1]
    adv_gini = Gini_index.gini(abs(adv_gs_np))
    adv_gs_np = adv_gs_np/np.max(adv_gs_np)
    adv_gs_np = adv_gs_np[adv_gs_np>0]
    plt.fill_between( np.arange(len(adv_gs_np)), 0, adv_gs_np, color = "r", alpha = 0.7,  label="Adv. trained")
    plt.xticks([])
    plt.ylabel('Significance',fontweight='bold',fontsize = 12)
    # plt.xlabel('Features (RGB)')
    plt.title('Noramlized Features Attribution Distribution.', fontweight='bold', y = -0.2, fontsize = 14)
    
    plt.legend(fontsize = 12, loc='center right')

    # dir_path = os.path.join(result_dir, str(img_id))
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    plt.savefig(os.path.join(result_dir, "normalized_feature_gini.png"))
    plt.close()
    
def draw_3d_importance_distribution(img, natural_img_gradshap, img_id, img_path):
     
    feature_map_mean = np.max(natural_img_gradshap.cpu().detach().numpy()[0], axis = 0)
    feature_map_positive = (feature_map_mean>0)*feature_map_mean
    # Normalize
    # feature_map_positive =  feature_map_positive/np.max(feature_map_positive)
    ax = plt.figure().add_subplot(projection='3d')
    X = np.arange(feature_map_mean.shape[0])
    Y = np.arange(feature_map_mean.shape[1])
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, feature_map_mean, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # img_arr = np.moveaxis(img_denorm(img)[0].cpu().detach().numpy(), 0, -1)



    # stride args allows to determine image quality 
    # stride = 1 work slow
    # ax.plot_surface(X, Y, np.atleast_2d(-0.01), rstride=1, cstride=1, facecolors=img_arr)

    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # alpha = 0.9
    # surf.set_facecolor((0, 0, 1, alpha))
    # Add a color bar which maps values to colors.
    plt.colorbar(surf, shrink=0.5, aspect=5)
    
    # ax.set_yticks([])
    # ax.set_xticks([])
    ax.set_zlim(-0.01, np.max(feature_map_positive))
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.view_init(60, 35)
    # dir_path = os.path.join(result_dir, str(img_id))
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    # plt.savefig(os.path.join(dir_path, "3d_feature_gini.png"))
    plt.savefig(img_path)
    plt.close()
    
def draw_org_img(input_img, img_path):
    plt.figure(figsize = (6,6), frameon=False)
    img = np.moveaxis(img_denorm(input_img)[0].cpu().detach().numpy(), 0, -1)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    # dir_path = os.path.join(result_dir, str(img_id))
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    # plt.savefig(os.path.join(result_dir, "{}.png".format(str(img_id))))
    plt.savefig(img_path, bbox_inches = 'tight', pad_inches= 0.05)
    plt.close()
    
def draw_occ_img(input_img, title_name, img_path):
    plt.figure(figsize = (6,6))
    img = np.moveaxis(img_denorm(input_img)[0].cpu().detach().numpy(), 0, -1)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.title(title_name,y = -0.1, fontweight="bold", fontsize = 14)
    # dir_path = os.path.join(result_dir, str(img_id))
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    # plt.savefig(os.path.join(result_dir, "{}.png".format(str(img_id))))
    plt.savefig(img_path)
    plt.close()
    
def draw_mask(input_mask, img_path):
    plt.figure(figsize = (6,6))
    img = np.moveaxis(input_mask[0].cpu().detach().numpy(), 0, -1)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    # dir_path = os.path.join(result_dir, str(img_id))
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    # plt.savefig(os.path.join(result_dir, "{}.png".format(str(img_id))))
    plt.savefig(img_path)
    plt.close()
    
def draw_feature_map(feature_map, img_id, img_path):
    plt.figure(figsize = (6,6))
    feature_map = np.moveaxis(img_denorm(feature_map)[0].cpu().detach().numpy(), 0, -1)
    feature_map = (feature_map > 0) * feature_map
    feature_map = (feature_map/np.max(feature_map))*255.0
    plt.imshow(feature_map)
    plt.yticks([])
    plt.xticks([])
    # dir_path = os.path.join(result_dir, str(img_id))
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    # plt.savefig(os.path.join(result_dir, "{}.png".format(str(img_id))))
    plt.savefig(img_path)
    plt.close()