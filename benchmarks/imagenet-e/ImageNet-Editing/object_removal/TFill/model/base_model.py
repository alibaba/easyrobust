import os, ntpath
import torch
from collections import OrderedDict
from util import util
from . import base_function
from abc import abstractmethod


class BaseModel():
    """This class is an abstract base class for models"""
    def __init__(self, opt):
        """Initialize the BaseModel class"""
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda') if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.value_names = []
        self.image_paths = []
        self.optimizers = []
        self.schedulers = []
        self.metric = 0  # used for learning rate policy 'plateau'

    def name(self):
        return 'BaseModel'

    @staticmethod
    def modify_options(parser, is_train):
        """Add new options and rewrite default values for existing options"""
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps"""
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load networks, create schedulers"""
        if self.isTrain:
            self.schedulers = [base_function.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = '%d' % opt.which_iter if opt.which_iter > 0 else opt.epoch
            self.load_networks(load_suffix)

        self.print_networks()

    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.to(self.device)
                if len(self.opt.gpu_ids) > 0:
                    setattr(self, 'net' + name, torch.nn.parallel.DataParallel(net, self.opt.gpu_ids))

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def log_imgs(self):
        """visualize the image during the training"""
        pass

    def test(self):
        """Forward function used in test time"""
        with torch.no_grad():
            self.forward()

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_losses(self):
        """Return training loss"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                try:
                    errors_ret[name] = float(getattr(self, 'loss_' + name))
                except:
                    pass
        return errors_ret

    def get_current_visuals(self):
        """Return visualization examples"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    visual_ret[name] = value[-1]
                else:
                    visual_ret[name] = value
        return visual_ret

    def save_networks(self, epoch, save_path=None):
        """Save all the networks to the disk."""
        save_path = save_path if save_path!= None else self.save_dir
        for name in self.model_names:
            if isinstance(name, str):
                filename = '%s_net_%s.pth' % (epoch, name)
                path = os.path.join(save_path, filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), path)

    def load_networks(self, epoch, save_path=None):
        """Load all the networks from the disk"""
        save_path = save_path if save_path != None else self.save_dir
        for name in self.model_names:
            if isinstance(name, str):
                filename = '%s_net_%s.pth' % (epoch, name)
                path = os.path.join(save_path, filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % path)
                try:
                    state_dict = torch.load(path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    net.load_state_dict(state_dict)
                except:
                    print('Pretrained network %s is unmatched' % name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()

    def print_networks(self):
        """Print the total number of parameters in the network and (if verbose) network architecture"""

        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_results(self, save_data, path=None, data_name='none'):
        """save the training or testing results to disk"""
        img_paths = self.get_image_paths()
        for i in range(save_data.size(0)):
            short_path = ntpath.basename(img_paths[i])  # get image path
            name = os.path.splitext(short_path)[0]
            img_name = '%s_%s.png' % (name, data_name)
            util.mkdir(path)
            img_path = os.path.join(path, img_name)
            img_numpy = util.tensor2im(save_data[i].unsqueeze(0))
            util.save_image(img_numpy, img_path)