import math
import os
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import wandb
from metrics.accuracy import Accuracy
from torch.utils.data import DataLoader
from torchvision.models import densenet, inception, resnet
from torchvision.transforms import (ColorJitter, RandomCrop,
                                    RandomHorizontalFlip, Resize)
from tqdm import tqdm

from models.base_model import BaseModel

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return torch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = torch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return torch.clamp(new_x, 0, 1)
    
class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError
        
class Classifier(BaseModel):
    def __init__(self,
                 num_classes,
                 in_channels=3,
                 architecture='resnet18',
                 pretrained=False,
                 name='Classifier',
                 *args,
                 **kwargs):
        super().__init__(name, *args, **kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.model = self._build_model(architecture, pretrained)
        #self.device = 'cuda:0'
        self.model.to(self.device)
        self.architecture = architecture

        self.to(self.device)

    def _build_model(self, architecture, pretrained):
        architecture = architecture.lower().replace('-', '').strip()
        if 'resnet' in architecture:
            
            if architecture == 'resnet18':
                model = resnet.resnet18(pretrained=pretrained)
            elif architecture == 'resnet34':
                model = resnet.resnet34(pretrained=pretrained)
            elif architecture == 'resnet50':
                model = resnet.resnet50(pretrained=pretrained)
            elif architecture == 'resnet101':
                model = resnet.resnet101(pretrained=pretrained)
            elif architecture == 'resnet152':
                model = resnet.resnet152(pretrained=pretrained)
            else:
                raise RuntimeError(
                    f'No RationalResNet with the name {architecture} available'
                )

            if self.num_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)

            return model

        elif 'resnext' in architecture:
            if architecture == 'resnext50':
                model = torch.hub.load('pytorch/vision:v0.6.0',
                                       'resnext50_32x4d',
                                       pretrained=pretrained)
            elif architecture == 'resnext101':
                model = torch.hub.load('pytorch/vision:v0.6.0',
                                       'resnext101_32x8d',
                                       pretrained=pretrained)
            else:
                raise RuntimeError(
                    f'No ResNext with the name {architecture} available')

            if self.num_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)

            return model

        elif 'resnest' in architecture:
            if architecture == 'resnest50':
                model = torch.hub.load('zhanghang1989/ResNeSt',
                                       'resnest50',
                                       pretrained=True)
            elif architecture == 'resnest101':
                from resnest.torch import resnest101
                model = resnest101(pretrained=True)
               
            elif architecture == 'resnest200':
                model = torch.hub.load('zhanghang1989/ResNeSt',
                                       'resnest200',
                                       pretrained=True)
            elif architecture == 'resnest269':
                model = torch.hub.load('zhanghang1989/ResNeSt',
                                       'resnest269',
                                       pretrained=True)
            else:
                raise RuntimeError(
                    f'No ResNeSt with the name {architecture} available')
            
            if self.num_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)

            return model

        elif 'densenet' in architecture:
            if architecture == 'densenet121':
                model = densenet.densenet121(pretrained=pretrained)
            elif architecture == 'densenet161':
                model = densenet.densenet161(pretrained=pretrained)
            elif architecture == 'densenet169':
                model = densenet.densenet169(pretrained=pretrained)
            elif architecture == 'densenet201':
                model = densenet.densenet201(pretrained=pretrained)
            else:
                raise RuntimeError(
                    f'No DenseNet with the name {architecture} available')

            if self.num_classes != model.classifier.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.classifier = nn.Linear(model.classifier.in_features,
                                             self.num_classes)
            return model

        # Note: inception_v3 expects input tensors with a size of N x 3 x 299 x 299, aux_logits are used per default
        elif 'inception' in architecture:
            model = inception.inception_v3(pretrained=pretrained,
                                           aux_logits=True,
                                           init_weights=True)
            if self.num_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            return model

        else:
            raise RuntimeError(
                f'No network with the name {architecture} available')

    def forward(self, x):
        
        if type(x) is np.ndarray:
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        out = self.model(x)
        return out
    
    def feature_forward(self, x):
        if type(x) is np.ndarray:
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        out = self.model(x)
        return out
    def adv_generator(self, images, target, eps, attack_steps, attack_lr, random_start, gpu, attack_criterion='regular', use_best=True, criterion=None):
        # generate adversarial examples
        prev_training = bool(self.model.training)
        self.model.eval()
        orig_input = images.detach().cuda(gpu, non_blocking=True)
        step = LinfStep(eps=eps, orig_input=orig_input, step_size=attack_lr)

        attack_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        best_loss = None
        best_x = None
        if random_start:
            images = step.random_perturb(images) 
        for _ in range(attack_steps):
            images = images.clone().detach().requires_grad_(True)
            adv_losses = -1 * criterion(self.model(images), target)

            if 0:
                with amp.scale_loss(torch.mean(adv_losses), []) as sl:
                    sl.backward()
            else:
                torch.mean(adv_losses).backward()
            grad = images.grad.detach()

            with torch.no_grad():
                varlist = [adv_losses, best_loss, images, best_x]
                best_loss, best_x = replace_best(*varlist) if use_best else (adv_losses, images)

                images = step.step(images, grad)
                images = step.project(images)

        adv_losses = criterion(self.model(images), target)
        varlist = [adv_losses, best_loss, images, best_x]
        best_loss, best_x = replace_best(*varlist) if use_best else (adv_losses, images)
        if prev_training:
            self.train()

        return best_x
    def adv_generator(self, images, target, eps, attack_steps, attack_lr, random_start, gpu, attack_criterion='regular', use_best=True, criterion=None):
        # generate adversarial examples
        prev_training = bool(self.model.training)
        self.model.eval()
        eps=0.3
        alpha=2/255
        ori_images = images.detach().cuda(gpu, non_blocking=True)
        
        for i in range(attack_steps) :    
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = -1 * criterion(outputs, target).to(self.device)
            cost.backward()
            adv_images = images + alpha*images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

        if prev_training:
            self.train()
        return images
    

    def fit(self,
            training_data,
            validation_data=None,
            test_data=None,
            optimizer=None,
            lr_scheduler=None,
            criterion=nn.CrossEntropyLoss(),
            metric=Accuracy,
            rtpt=None,
            config=None,
            batch_size=64,
            num_epochs=30,
            dataloader_num_workers=8,
            enable_logging=False,
            wandb_init_args=None,
            save_base_path="",
            config_file=None):

        trainloader = DataLoader(training_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=dataloader_num_workers,
                                 pin_memory=True)

        if rtpt is None:
            print('Please use RTPT (Remaining Time to Process Title)')

        # Initialize WandB logging
        if enable_logging:

            if wandb_init_args is None:
                wandb_init_args = dict()

            wandb_config = {
                "Dataset": config.dataset['type'],
                'Epochs': num_epochs,
                'Batch_size': batch_size,
                'Initial_lr': optimizer.param_groups[0]['lr'],
                'Architecture': self.architecture,
                'Pretrained': self.pretrained,
                'Optimizer': optimizer,
                'Trainingset_size': len(training_data),
                'num_classes': self.num_classes,
                'Total_parameters':
                self.count_parameters(only_trainable=False),
                'Trainable_parameters':
                self.count_parameters(only_trainable=True)
            }

            for t in training_data.transform.transforms:
                if type(t) is Resize:
                    wandb_config['Resize'] = t.size
                elif type(t) is RandomCrop:
                    wandb_config['RandomCrop'] = t.size
                elif type(t) is ColorJitter:
                    wandb_config['BrightnessJitter'] = t.brightness
                    wandb_config['ContrastJitter'] = t.contrast
                    wandb_config['SaturationJitter'] = t.saturation
                    wandb_config['HueJitter'] = t.hue
                elif type(t) is RandomHorizontalFlip:
                    wandb_config['HorizontalFlip'] = t.p

            if validation_data:
                wandb_config['Validationset_size'] = len(validation_data)

            if test_data:
                wandb_config['Testset_size'] = len(test_data)

            wandb.init(**wandb_init_args, config=wandb_config, reinit=True)
            wandb.watch(self.model)
            if config_file:
                wandb.save(config_file)

        # Training cycle
        best_model_values = {
            'validation_metric': 0.0,
            'validation_loss': float('inf'),
            'model_state_dict': None,
            'model_optimizer_state_dict': None,
            'training_metric': 0,
            'training_loss': 0,
        }

        metric_train = metric()

        print('----------------------- START TRAINING -----------------------')
        #num_epochs = 1
        for epoch in range(num_epochs):
            # Training
            print(f'Epoch {epoch + 1}/{num_epochs}')
            with torch.cuda.device('cuda:0'):
                torch.cuda.empty_cache()
            running_total_loss = 0.0
            running_main_loss = 0.0
            running_aux_loss = 0.0
            metric_train.reset()
            self.train()
            self.to(self.device)
            tmp = 0
            for inputs, labels in tqdm(trainloader,
                                       desc='training',
                                       leave=False,
                                       file=sys.stdout):
                with torch.cuda.device('cuda:7'):
                    torch.cuda.empty_cache()
                inputs, labels = inputs.to(self.device), labels.to(
                                               self.device)
                #while True:
                    #attack_label = torch.randint(0,1000,(labels.size(0),)).to(self.device)
                    #if torch.sum(attack_label == labels).item() == 0:
                        #break
                #if tmp % 2 ==0:
                    #inputs = self.adv_generator(inputs, labels, 0.01/255, 1, 0.1/255/2, random_start=True, gpu=self.device, use_best=False, attack_criterion='regular',criterion=criterion)
                tmp += 1
                model_output = self.forward(inputs)
                #model_output = self.forward(inputs)
                aux_loss = torch.tensor(0.0, device=self.device)

                # Separate Inception_v3 outputs
                aux_logits = None
                if isinstance(model_output, inception.InceptionOutputs):
                    if self.model.aux_logits:
                        model_output, aux_logits = model_output

                main_loss = criterion(model_output, labels) 
                
                if aux_logits is not None:
                    aux_loss += criterion(aux_logits, labels).sum()

                num_samples = inputs.shape[0]
                loss = main_loss + aux_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_total_loss += loss * num_samples
                running_main_loss += main_loss * num_samples
                running_aux_loss += aux_loss * num_samples
                del loss, main_loss, aux_loss
                metric_train.update(model_output, labels)

            print(
                f'Training {metric_train.name}:   {metric_train.compute_metric():.2%}',
                f'\t Epoch total loss: {running_total_loss / len(training_data):.4f}',
                f'\t Epoch main loss: {running_main_loss / len(training_data):.4f}',
                f'\t Aux loss: {running_aux_loss / len(training_data):.4f}')

            if enable_logging:
                wandb.log(
                    {
                        f'Training {metric_train.name}':
                        metric_train.compute_metric(),
                        'Training Loss':
                        running_total_loss / len(training_data),
                    },
                    step=epoch)

            # Validation
            if validation_data:
                self.eval()
                val_metric, val_loss = self.evaluate(
                    validation_data,
                    batch_size,
                    metric,
                    criterion,
                    dataloader_num_workers=dataloader_num_workers)

                print(
                    f'Validation {metric_train.name}: {val_metric:.2%} \t Validation Loss:  {val_loss:.4f}'
                )

                # Save best model
                if val_metric > best_model_values['validation_metric']:
                    print('Copying better model')
                    best_model_values['validation_metric'] = val_metric
                    best_model_values['validation_loss'] = val_loss
                    best_model_values['model_state_dict'] = deepcopy(
                        self.state_dict())
                    best_model_values['model_optimizer_state_dict'] = deepcopy(
                        optimizer.state_dict())
                    best_model_values[
                        'training_metric'] = metric_train.compute_metric()
                    best_model_values[
                        'training_loss'] = running_total_loss / len(
                            trainloader)

                if enable_logging:
                    wandb.log(
                        {
                            f'Validation {metric_train.name}': val_metric,
                            'Validation Loss': val_loss,
                        },
                        step=epoch)
            else:
                best_model_values['validation_metric'] = None
                best_model_values['validation_loss'] = None
                best_model_values['model_state_dict'] = deepcopy(
                    self.state_dict())
                best_model_values['model_optimizer_state_dict'] = deepcopy(
                    optimizer.state_dict())
                best_model_values[
                    'training_metric'] = metric_train.compute_metric()
                best_model_values['training_loss'] = running_total_loss / len(
                    trainloader)

            # Update the RTPT
            rtpt.step(
                subtitle=f"loss={running_total_loss / len(trainloader):.4f}")

            # make the lr scheduler step
            if lr_scheduler is not None:
                lr_scheduler.step()

        # save the final model
        if validation_data:
            self.load_state_dict(best_model_values['model_state_dict'])

        if save_base_path:
            if not os.path.exists(save_base_path):
                os.makedirs(save_base_path)
            if validation_data:
                model_path = os.path.join(
                    save_base_path, self.name +
                    f'_{best_model_values["validation_metric"]:.4f}' + '.pth')
            else:
                model_path = os.path.join(
                    save_base_path, self.name +
                    f'_{best_model_values["training_metric"]:.4f}_no_val' +
                    '.pth')

        else:
            model_path = self.name

        torch.save(
            {
                'epoch':
                num_epochs,
                'model_state_dict':
                best_model_values['model_state_dict'],
                'optimizer_state_dict':
                best_model_values['model_optimizer_state_dict'],
            }, model_path)

        # Test final model
        test_metric, test_loss = None, None
        if test_data:
            test_metric, test_loss = self.evaluate(
                test_data,
                batch_size,
                metric,
                criterion,
                dataloader_num_workers=dataloader_num_workers)
            print(
                '----------------------- FINISH TRAINING -----------------------'
            )
            print(
                f'Final Test {metric_train.name}: {test_metric:.2%} \t Test Loss: {test_loss:.4f} \n'
            )

        if enable_logging:
            wandb.save(model_path)
            wandb.run.summary[
                f'Validation {metric_train.name}'] = best_model_values[
                    'validation_metric']
            wandb.run.summary['Validation Loss'] = best_model_values[
                'validation_loss']
            wandb.run.summary[
                f'Training {metric_train.name}'] = best_model_values[
                    'training_metric']
            wandb.run.summary['Training Loss'] = best_model_values[
                'training_loss']
            wandb.run.summary[f'Test {metric_train.name}'] = test_metric
            wandb.run.summary['Test Loss'] = test_loss

            wandb.config.update({'model_path': model_path})
            wandb.config.update({'config_path': config_file})
            wandb.finish()

    def evaluate(self,
                 evaluation_data,
                 batch_size=128,
                 metric=Accuracy,
                 criterion=nn.CrossEntropyLoss(),
                 dataloader_num_workers=4):
        evalloader = DataLoader(evaluation_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=dataloader_num_workers,
                                pin_memory=True)
        metric = metric()
        self.eval()
        with torch.no_grad():
            running_loss = torch.tensor(0.0, device=self.device)
            for inputs, labels in tqdm(evalloader,
                                       desc='Evaluating',
                                       leave=False,
                                       file=sys.stdout):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                model_output = self.forward(inputs)
                metric.update(model_output, labels)
                running_loss += criterion(model_output,
                                          labels).cpu() * inputs.shape[0]

            metric_result = metric.compute_metric()

            return metric_result, running_loss.item() / len(evaluation_data)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def unfreeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()
