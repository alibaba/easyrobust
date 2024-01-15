import logging
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.functional import normalize, linear
from torch.nn.parameter import Parameter
import copy

class DMMIA(Module):
    """s
    Modified from Partial-FC
    """

    @torch.no_grad()
    def __init__(self, rank, local_rank, world_size, batch_size, resume,
                 margin_softmax, num_classes, sample_rate=1.0, embedding_size=512, prefix="./", cfg=None):
        super(VPL, self).__init__()
        #
        assert sample_rate==1.0
        assert not resume
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.local_rank: int = local_rank
        self.device: torch.device = torch.device("cuda:{}".format(self.local_rank))
        self.world_size: int = world_size
        self.batch_size: int = batch_size
        self.margin_softmax: callable = margin_softmax
        self.sample_rate: float = sample_rate
        self.embedding_size: int = embedding_size
        self.prefix: str = prefix
        world_size = 1
        rank = 0
        self.num_local: int = num_classes // world_size + int(rank < num_classes % world_size)
        self.class_start: int = num_classes // world_size * rank + min(rank, num_classes % world_size)
        self.class_start = 1000
        self.num_sample: int = int(self.sample_rate * self.num_local)
        
        self.device = 'cuda:0'
        self.weight_name = "./rank_softmax_weight.pt"
        self.weight_mom_name = "./rank_softmax_weight_mom.pt"
        self.world_size = 1
        self.vpl_t = 800
        self.weight = torch.normal(0, 0.01, (1000, self.vpl_t,self.embedding_size), device=self.device)
        self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
 
        
        logging.info("softmax weight init successfully!")
        logging.info("softmax weight mom init successfully!")
        self.stream: torch.cuda.Stream = torch.cuda.Stream(local_rank)

        self.index = None
        self.sub_weight = Parameter(self.weight)
        self.sub_weight_mom = self.weight_mom

        #vpl variables

        self._iters = 0
        self.cfg = cfg
        self.cfg =  {'start_iters': 8000, 'allowed_delta': 200, 'lambda': 0.15, 'mode': -1, 'momentum': False} #mode==-1 disables vpl
        self.vpl_mode = 1
        if self.cfg is not None:
            self.vpl_mode = self.cfg['mode']
            self.vpl_mode = 0
            N_m = 1
            self.N_m = N_m
            self.vpl_t_ = 1000
            if self.vpl_mode>=0:
                self.register_buffer("queue", torch.randn(self.vpl_t_, N_m, self.embedding_size, device=self.device))
                self.queue = normalize(self.queue)
                self.register_buffer("queue_iters", torch.zeros((self.vpl_t_,), dtype=torch.long, device=self.device))
                self.register_buffer("queue_lambda", torch.zeros((self.vpl_t_,), dtype=torch.float32, device=self.device))

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)
    def save_params(self):
        pass

    @torch.no_grad()
    def sample(self, total_label):
       
        index_positive =  (total_label < self.class_start + self.num_local)
        total_label = total_label
        return index_positive

    def forward(self, total_features, norm_weight):
        torch.cuda.current_stream().wait_stream(self.stream)
        logits = linear(total_features, norm_weight)
        return logits
    def forward_18_n(self, x, norm_weight, total_label):
        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.weight.T,2),0, keepdim=True)
        
        features_into_centers = 2 * torch.matmul(x, (self.weight[total_label][:, :400, :].reshape(self.weight[total_label].shape[0]* self.weight[total_label][:, :400, :].shape[1], self.weight[total_label][:, :400, :].shape[-1]).T))
        dist = features_into_centers
        return self.weight[total_label][0].T, -dist
    def forward_18_n_2(self, x, norm_weight, total_label):
        return norm_weight, 0

    def forward_18(self, x, norm_weight, total_label):
        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.weight.T,2),0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.weight[total_label][0].T))

        dist = features_into_centers
        return self.weight[total_label][0].T, -dist

    @torch.no_grad()
    def update(self):
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        with torch.cuda.stream(self.stream):
            total_label = torch.zeros(
                size=[self.batch_size * self.world_size], device=self.device, dtype=torch.long)
            total_label = label.clone() 
            index_positive = self.sample(total_label)
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = normalize(self.sub_weight)
            return total_label, norm_weight, index_positive

    @torch.no_grad()
    def prepare_queue_lambda(self, label, iters):
        self.queue_lambda[:] = 0.0
        if iters>self.cfg['start_iters']:
            allowed_delta = self.cfg['allowed_delta']
            if self.vpl_mode==0:
                past_iters = iters - self.queue_iters
                idx = torch.where(past_iters <= allowed_delta)[0]
                self.queue_lambda[idx] = self.cfg['lambda']

            if iters % 2000 == 0 and self.rank == 0:
                logging.info('[%d]use-lambda: %d/%d'%(iters,len(idx), self.num_local))

    @torch.no_grad()
    def set_queue(self, total_features, total_label, index_positive, iters, pre_idx):
        local_label = total_label[index_positive]
        sel_features = normalize(total_features[index_positive,:])
        pre_idx = pre_idx.squeeze(-1)
        m = 0.7
        for i in range(total_label.shape[0]):
            if self.queue_iters[total_label[i]] == 0:
                self.queue[total_label[i], 0, :] = sel_features[i]
                self.queue_iters[total_label[i]] += 1
            else:
                self.queue[total_label[i], 0, :] = m * self.queue[total_label[i], 0, :] + (1-m) * sel_features[i]

    def regularization(self, features, centers, labels):
        batch_size = features.shape[0] 
        dist_1 = 0
        dist_2 = 0
        dist_1 = torch.matmul(features,centers[:, :400])
        dist_1 = torch.mean(torch.exp(dist_1), dim=1).sum()
        
        dist_2 = torch.matmul(features,centers[:, 400:])
        dist_2 = torch.mean(torch.exp(dist_2), dim=1).sum()
        dist = dist_1 / (dist_1 +dist_2)
        loss = (dist.clamp(min=1e-12, max=1e+12).sum() / batch_size)
        return loss
    def regularization_0(self, features, centers, labels):
        distance=(features-torch.t(centers)[labels])

        distance=torch.sum(torch.pow(distance,2),1, keepdim=True)

        distance=(torch.sum(distance, 0, keepdim=True))/features.shape[0]

        return distance
    def regularization_0_0(self, features, centers, labels):

        batch_size = features.shape[0] 
        dist_1 = 0
        dist_2 = 0
        dist = 0
        loss = 0
        labels = labels.cpu()
        for i in range(batch_size):
            p_centers = centers[labels[i]]

            lab_fake_0 = torch.arange(labels[i])
            lab_fake_1 = torch.arange(1000)
            lab_fake = torch.cat((lab_fake_0, lab_fake_1[labels[i]+1:]), 0)
            n_centers = centers[lab_fake] 
            dist_1 = torch.matmul(features[i],p_centers.T)
            dist_1 = torch.mean(torch.exp(dist_1), dim=0).sum()
            dist_2 = torch.matmul(features,n_centers.reshape(999*self.N_m, 512).T)
            dist_2 = torch.mean(torch.exp(dist_2), dim=1).sum()
            dist = dist_1 / (dist_1 +dist_2)
            loss += (dist.clamp(min=1e-12, max=1e+12).sum() / batch_size)

        return loss
    def forward_backward(self, label, features, imgs, optimizer, target_model, outputs=None, feature_w=None, label_fake=None):
        self._iters += 1
        device_i = features.device
        total_label, norm_weight, index_positive = self.prepare(label, optimizer)
        total_label = label.clone()

        features = torch.nn.functional.normalize(features)
        total_features = features.data
        total_features.requires_grad = True
        y_pred = torch.topk(outputs, dim=1, k=1).indices
        if feature_w is not None:
            total_feature_w = torch.zeros(
                size=[self.batch_size * self.world_size, self.embedding_size], device=device_i)
            dist.all_gather(list(total_feature_w.chunk(self.world_size, dim=0)), feature_w.data)
   
        if self.vpl_mode>=0:
            injected_weight_1 = norm_weight
            injected_weight_2 = self.queue
            injected_norm_weight_1 = normalize(injected_weight_1)
            injected_norm_weight_2 = normalize(injected_weight_2)
            centers_1, x_1 = self.forward_18_n(total_features, injected_norm_weight_1, total_label)
            centers_2, x_2 = self.forward_18_n_2(total_features, injected_norm_weight_2, total_label)
        else:
            norm_weight = norm_weight.to(device_i)
            logits = self.forward(total_features, norm_weight)
        
        loss1=self.regularization(total_features, centers_1, total_label)
        loss2 = self.regularization_0_0(total_features, centers_2, total_label)
        loss_t = 1 * loss1 + 1 * loss2
        loss_t.backward() 
        if total_features.grad is not None:
            total_features.grad.detach_()
        
        x_grad: torch.Tensor = torch.zeros_like(features, requires_grad=True)
        x_grad = total_features.grad
        x_grad = x_grad * self.world_size
        
        if self.vpl_mode>=0:
            if feature_w is None:
                self.set_queue(total_features.detach(), total_label, index_positive, self._iters, y_pred)
            else:
                self.set_queue(total_feature_w, total_label, index_positive, self._iters, y_pred)
        return x_grad, loss_t

