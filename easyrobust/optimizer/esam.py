"""
@inproceedings{du2021efficient,
  title={Efficient Sharpness-aware Minimization for Improved Training of Neural Networks},
  author={Du, Jiawei and Yan, Hanshu and Feng, Jiashi and Zhou, Joey Tianyi and Zhen, Liangli and Goh, Rick Siow Mong and Tan, Vincent},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
"""

import torch
import torch.nn.functional as F
import random

class ESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,beta=1.0,gamma=1.0,adaptive=False,**kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.beta = beta
        self.gamma = gamma

        defaults = dict(rho=rho,adaptive=adaptive, **kwargs)
        super(ESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        #first order sum 
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / self.beta
            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None: continue
                #original sam 
                # e_w = p.grad * scale.to(p)
                #asam 
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 1)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    '''
    @torch.no_grad()
    def first_half(self, zero_grad=False):
        #first order sum 
        for group in self.param_groups:
            for p in group["params"]:
                if self.state[p]:
                    p.add_(self.state[p]["e_w"]*0.90)  # climb to the local maximum "w + e(w)"
    '''


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                self.state[p]["e_w"] = 0

                if random.random() > self.beta:
                    p.requires_grad = False

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self):
        inputs,targets,loss_fct,model,defined_backward = self.paras
        assert defined_backward is not None, "Sharpness Aware Minimization requires defined_backward, but it was not provided"

        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True


        logits = model(inputs)
        loss = loss_fct(logits,targets)

        l_before = loss.clone().detach()
        predictions = logits
        return_loss = loss.clone().detach()
        loss = loss.mean()
        defined_backward(loss)

        #first step to w + e(w)
        self.first_step(True)


        with torch.no_grad():
            l_after = loss_fct(model(inputs),targets)
            instance_sharpness = l_after-l_before

            #codes for sorting 
            prob = self.gamma
            if prob >=0.99:
                indices = range(len(targets))
            else:
                position = int(len(targets) * prob)
                cutoff,_ = torch.topk(instance_sharpness,position)
                cutoff = cutoff[-1]

                # cutoff = 0
                #select top k% 

                indices = [instance_sharpness > cutoff] 


        # second forward-backward step
        # self.first_half()

        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False



        loss = loss_fct(model(inputs[indices]), targets[indices])
        loss = loss.mean()
        defined_backward(loss)
        self.second_step(True)

        self.returnthings = (predictions,return_loss)
 

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        #original sam 
                        # p.grad.norm(p=2).to(shared_device)
                        #asam 
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm