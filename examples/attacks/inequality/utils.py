import torchvision.datasets as dsets
import torch
from captum.attr import GradientShap,IntegratedGradients,NoiseTunnel, InputXGradient
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def img_denorm(x, mean= [0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def image_folder_custom_label(root, transform, idx2label) :
    
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                 target_transform=lambda x : idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data



def get_pred(net, img, label):
    output = net(img)
    _, pred = torch.max(output, dim=1)
    pecent_all = torch.nn.functional.softmax(output, dim=1)[0] * 100
    percentage = pecent_all[int(label)].item()
    correct = (pred==label).cpu().detach().numpy()
    return pred, correct




def get_gradshap(net, test_input, test_target, method = "ig"):
    avg_attr_gs = torch.zeros_like(test_input)

    input = test_input.to(device)
    input.required_grad = True
    rand_img_dist = torch.cat([input * 0, input * 1])    
    if method == "ig":
        ig = IntegratedGradients(net)
        attributions_gs=ig.attribute(input,n_steps=50,baselines=input*0, target=test_target.to(device), return_convergence_delta=False)
    elif method == "nt":
        ig = IntegratedGradients(net)  
        nt = NoiseTunnel(ig)
        attributions_gs = nt.attribute(input, baselines=input * 0, nt_type='smoothgrad_sq', target=test_target.to(device), 
                                          nt_samples=20, nt_samples_batch_size = 1, stdevs=0.2)
    elif method == "gs":
        gradient_shap = GradientShap(net)
        attributions_gs = gradient_shap.attribute(input,n_samples=50,stdevs=0.01,baselines=rand_img_dist, target=test_target.to(device))
    elif method == "grad":
        input_x_gradient = InputXGradient(net)
        attributions_gs = input_x_gradient.attribute(input, target=test_target.to(device))
    return attributions_gs