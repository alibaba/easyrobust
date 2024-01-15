from losses.poincare import poincare_loss
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
from torch.autograd import grad
import random
import math
import utils
from torchvision import transforms
from torch.nn.utils import clip_grad_norm_
from torch.nn.modules.container import Sequential
class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]
    
def gkern(kernlen = 3, nsig = 1):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(-1,x)
    kern2d = st.norm.pdf(-1,x)
    kernel_raw = np.outer(kern1d, kern2d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def blur(tensor_image, epsilon, stack_kerne):        
    min_batch=tensor_image.shape[0]        
    channels=tensor_image.shape[1]        
    out_channel=channels       
    kernel=torch.FloatTensor(stack_kerne).cuda()	
    weight = nn.Parameter(data=kernel, requires_grad=False)         
    data_grad=F.conv2d(tensor_image,weight,bias=None,stride=1,padding=(2,0), dilation=2)

    sign_data_grad = data_grad.sign()
	
    perturbed_image = tensor_image + epsilon*sign_data_grad
    return data_grad * epsilon
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers= extracted_layers
 
    def forward(self, x):
        for name, module in self.submodule.module.named_modules():
            print(x.shape, '***')
            x = module(x)
        print(x.shape)
        exit()
        for _, module in self.submodule.named_children():
            print(module)
            if(isinstance (module, Sequential)):
                for ind_a in len(module):
                    x = module[ind_a](x)
            else:
                x = module(x)

        #for name, module in self.submodule.module.items():
            #print(name)
            #if name is not "fc": 
                #x = module(x)
        exit()
        return x


#from randomizations import StyleRandomization, ContentRandomization
class Optimization():
    def __init__(self, target_model, synthesis, discriminator, transformations, num_ws, config, vae=None, vae_optimizer=None, vae_scheduler=None):
        self.synthesis = synthesis
        self.target = target_model
        self.discriminator = discriminator
        self.config = config
        self.transformations = transformations
        self.discriminator_weight = self.config.attack['discriminator_loss_weight']
        self.num_ws = num_ws
        self.clip = config.attack['clip']
        #0, 16, 17, 18
        self.gaijin = 18
        if self.gaijin == 11:
            self.vae = vae[0]
            self.vae_optimizer = vae_optimizer[0]
            self.vae_scheduler = vae_scheduler[0]
            self.unet = vae[1]
            self.unet_optimizer = vae_optimizer[1]
            self.unet_scheduler = vae_scheduler[1]
        if self.gaijin == 13 or self.gaijin == 14 or self.gaijin == 15 or self.gaijin ==16 or self.gaijin == 17 or self.gaijin == 18:
            self.module_fc = vae
            self.opt_pfc = vae_optimizer
            self.scheduler_pfc = vae_scheduler
            
    
    def optimize(self, w_batch, targets_batch, num_epochs, device, w_batch_):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)
        
        loss_engry = 0
        # Start optimization
        lambda_list = []

        for i in range(num_epochs):
            
            # synthesize imagesnd preprocess images

            imgs = self.synthesize(w_batch, num_ws=self.num_ws) #16,3,1024,1024
            # compute discriminator loss
            
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs, fearure_w = self.target(imgs) #16,1000 y:16 [16, 512, 1, 1]
            feature_w = fearure_w.squeeze(-1).squeeze(-1)
            
            target_loss = poincare_loss(
                outputs, targets_batch).mean()
            loss_engry = 0
           
            if self.gaijin == 13:
                feature_w = None
                targets = targets_batch.clone()
                label_fake = torch.argmax(outputs, dim=1, keepdim=True).squeeze(-1)
                x_grad, loss_v = self.module_fc.forward_backward(targets, outputs, self.opt_pfc, feature_w, label_fake)
            if self.gaijin == 14:
                feature_w = None
                targets = targets_batch.clone()
                x_grad, loss_v = self.module_fc.forward_backward(targets, imgs, self.opt_pfc, self.target)
            if self.gaijin == 17:
                feature_w = feature_w
                #feature_w = outputs
                targets = targets_batch.clone()
                x_grad, loss_v = self.module_fc.forward_backward(targets, feature_w, imgs, self.opt_pfc, self.target)
            if self.gaijin == 18:
                targets = targets_batch.clone()
                x_grad, loss_v = self.module_fc.forward_backward(targets, feature_w, imgs, self.opt_pfc, self.target, outputs)
            if self.gaijin == 15 or self.gaijin==16:
                feature_w = None
                targets = targets_batch.clone()
                x_grad, loss_v = self.module_fc.forward_backward(targets, outputs, imgs, self.opt_pfc, self.target)
            if self.gaijin == 0:
                optimizer.zero_grad()
            if self.gaijin:
                loss = target_loss  + discriminator_loss * self.discriminator_weight 
            else:
                loss = target_loss + discriminator_loss * self.discriminator_weight 
            if self.gaijin == 0:
                loss.backward()
                optimizer.step()
            
            if self.gaijin==14:
                print('grad-----', x_grad.sum())
                imgs.backward(x_grad*0.000001)
                optimizer.step()
                self.opt_pfc.zero_grad()
                self.module_fc.update()
                self.opt_pfc.step()

                
            if self.gaijin==13 or self.gaijin==15:
                outputs.backward( x_grad*0.000001)
                #clip_grad_norm_(w_batch, max_norm=10, norm_type=2)
                optimizer.step()
                
                self.opt_pfc.zero_grad()
                self.module_fc.update()
                self.opt_pfc.step()
                optimizer.zero_grad()
            if self.gaijin == 16:
                optimizer.zero_grad()
                loss = target_loss
                outputs.retain_grad()
                loss.backward(retain_graph=True)
                outputs_grad = outputs.grad
                optimizer.step()
                '''
                grad_weight = 0
                for tt in range(4):
                    tmp = torch.cosine_similarity(x_grad[tt], outputs_grad[tt], dim=0)
                    print(tmp)
                    grad_weight += torch.cosine_similarity(x_grad[tt], outputs_grad[tt], dim=0)
                grad_weight = grad_weight/4.0
                '''
                grad_weight = torch.mean(torch.sum(x_grad * outputs_grad, dim=-1))
                #grad_weight = 0.08 * 0.000001
                outputs.backward(-1 * x_grad*grad_weight)
                optimizer.step()
                self.opt_pfc.zero_grad()
                self.module_fc.update()
                self.opt_pfc.step()
                optimizer.zero_grad()
            if self.gaijin == 17 or self.gaijin==18:
                optimizer.zero_grad()
                loss = target_loss
                feature_w.retain_grad()
                loss.backward(retain_graph=True)
                outputs_grad = feature_w.grad
                optimizer.step()
                
                #grad_weight = torch.mean(torch.sum(x_grad * outputs_grad, dim=-1))
                #feature_w.backward(-1 * x_grad*grad_weight)
                feature_w.backward( x_grad*1 * 0.000001) 
                optimizer.step()
                self.opt_pfc.zero_grad()
                self.module_fc.update()
                self.opt_pfc.step()
                optimizer.zero_grad()
            if scheduler:
                #scheduler.step()
                #self.vae_scheduler.step()
                #self.unet_scheduler.step()
                scheduler.step()
                self.scheduler_pfc.step()
            # Log results
            if self.config.log_progress:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(
                            confidence_vector.to(device), 1, targets_batch.unsqueeze(1).to(device))
                    
                    mean_conf = confidences.mean().detach().cpu()
                tmp = int(device.split(':')[-1])
              
                if torch.cuda.current_device() == 0 or torch.cuda.current_device() ==4:
                    print(
                        f'iteration {i}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} \t kl_loss={loss_engry:.4f}'
                    )
        return w_batch.detach()

    def synthesize(self, w, num_ws):
        if w.shape[1] == 1:
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=num_ws,
                                                 dim=1)
            #print(w_expanded.device, w_expanded.shape)
            imgs = self.synthesis(w_expanded,
                                  noise_mode='const',
                                  force_fp32=True)
        else:
            imgs = self.synthesis(w, noise_mode='const', force_fp32=True)
        return imgs

    def clip_images(self, imgs):
        lower_limit = torch.tensor(-1.0).float().to(imgs.device)
        upper_limit = torch.tensor(1.0).float().to(imgs.device)
        imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
        imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
        return imgs

    def compute_discriminator_loss(self, imgs):
        discriminator_logits = self.discriminator(imgs, None)
        discriminator_loss = nn.functional.softplus(
            -discriminator_logits).mean()
        return discriminator_loss
