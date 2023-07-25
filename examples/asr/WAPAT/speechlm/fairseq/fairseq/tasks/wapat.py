import random
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import torchaudio

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
        return torch.clamp(diff + self.orig_input, -1, 1)

    def step(self, x, g):
        """
        """
        step = torch.sign(g) * self.step_size
        return x + step
 
    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return torch.clamp(new_x, -1, 1)

class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return torch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)

def replace_best(loss, bloss, x, bx, m):
    if bloss is None:
        bx = x.clone().detach()
        bloss = loss.clone().detach()
    else:
        replace = m * bloss < m * loss
        bx[replace] = x[replace].clone().detach()
        bloss[replace] = loss[replace]

    return bloss, bx

def random_perturb_gege(x):
    new_x = x + 0.001 * torch.rand_like(x)
    return new_x

import gpuRIR
import numpy as np
class Parameter:
    def __init__(self, *args):
        if len(args) == 1:
            self.random = False
            self.value = np.array(args[0])
            self.min_value = None
            self.max_value = None
        elif len(args) == 2:
            self.random = True
            self.min_value = np.array(args[0])
            self.max_value = np.array(args[1])
            self.value = None
        else:
            raise Exception(
                'Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
    def getvalue(self):
        if self.random:
            return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
        else:
            return self.value

class GpuRirDemo:
    def __init__(self, room_sz, t60, beta, fs, array_pos):
        self.room_sz = room_sz
        self.t60 = t60
        self.beta = beta
        self.fs = fs
        self.array_pos = array_pos
    def simulate(self, y):
        if self.t60 == 0:
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1, 1, 1]
        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(15, self.t60)
            Tmax = gpuRIR.att2t_SabineEstimator(60, self.t60)
            if self.t60 < 0.15: Tdiff = Tmax
            nb_img = gpuRIR.t2n(Tdiff, self.room_sz)
        # mic position
        mic_pos = np.array(((-0.079, 0.000, 0.000),
                            (-0.079, -0.009, 0.000),
                            (0.079, 0.000, 0.000),
                            (0.079, -0.009, 0.000)))
        array_pos = self.array_pos * self.room_sz
        mic_pos = mic_pos + array_pos
        source_pos = np.random.rand(3) * self.room_sz
        RIR = gpuRIR.simulateRIR(
            room_sz=self.room_sz,
            beta=self.beta,
            nb_img=nb_img,
            fs=self.fs,
            pos_src=np.array([source_pos]),
            pos_rcv=mic_pos,
            Tmax=Tmax,
            Tdiff=Tdiff,
            mic_pattern='omni'
        )
        mic_sig = gpuRIR.simulateTrajectory(y.cpu(), RIR, fs=self.fs)
        return mic_sig[:,0]

class Aug_guide():
    def __init__(self) -> None:
        musan_path = './musan/musan.txt'
        self.musan = []
        with open(musan_path, 'r', encoding="utf-8" ) as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                lines = lines.strip()
                self.musan.append(lines)
    def aug_RIR(self, sample_room):
        test_code = GpuRirDemo(
        room_sz=Parameter([3, 3, 2.5], [4, 5, 3]).getvalue(), 
        t60=Parameter(0.2, 1.0).getvalue(), 
                        beta=Parameter([0.5]*6, [1.0]*6).getvalue(), 
                        array_pos=Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]).getvalue(),
                        fs=16000)
        device_tmp = sample_room['net_input']['source'].device

        y_new = sample_room['net_input']['source']
        sample_room['net_input']['source'] = []
        for idx in range(len(y_new)):
            with torch.no_grad():
                sample_room['net_input']['source'].append(test_code.simulate(y_new[idx])[:y_new.shape[1]])
        sample_room['net_input']['source'] = torch.tensor(sample_room['net_input']['source']).to(device_tmp).half()
        return sample_room
    def time_mask_augment(self, sample_room):
        inputs = sample_room['net_input']['source']

        time_len = inputs.shape[1]
        max_mask_time = torch.randint(0, 2000, (1,)).item()
        mask_num = torch.randint(0, 10, (1,)).item()
        for i in range(mask_num):
            t = np.random.uniform(low=0.0, high=max_mask_time)
            t = int(t)
            t0 = random.randint(0, time_len - t)
            inputs[:, t0:t0 + t] = 0
        sample_room['net_input']['source'] = inputs
        return sample_room
    def volume_augment(self, sample_room, min_gain_dBFS=-10, max_gain_dBFS=10):
        samples = sample_room['net_input']['source'].copy() 
        data_type = samples[0].dtype
        gain = np.uniform(min_gain_dBFS, max_gain_dBFS)
        gain = 10. ** (gain / 20.)
        samples = samples * torch.tensor(gain)
        samples = samples.astype(data_type)
        sample_room['net_input']['source'] = samples
        return sample_room

def sample_aug(sample_room):
    import random
    a = random.random()
    if a < 0.2:
        aug_guide = Aug_guide()
    elif a>=0.2 and a<=0.4:
        aug_guide = Aug_guide()
        musan_len = len(aug_guide.musan)
        key = random.randint(0,musan_len-1)
        noise_path = aug_guide.musan[key]
       
        device_tmp = sample_room['net_input']['source'].device
        noise_audio, sf = torchaudio.load(noise_path)
        noise_audio = noise_audio.repeat(sample_room['net_input']['source'].shape[0], 1)
        if noise_audio.shape[1] < sample_room['net_input']['source'].shape[1]:
            beishu = sample_room['net_input']['source'].shape[1] // noise_audio.shape[1]
            noise_audio = noise_audio.repeat(1, beishu + 1)
        noise_audio = noise_audio[:, :sample_room['net_input']['source'].shape[1]].to(device_tmp).type_as(sample_room['net_input']['source'])
        sample_room['net_input']['source'] =  0.3 * noise_audio + sample_room['net_input']['source']
    else:
        from fairseq.tasks.process_file import augmentation_factory
        speech = sample_room['net_input']['source']
        sampling_rate = 16000
        if a > 0.4 and a <= 0.6:
            description = 'time_drop'
        if a > 0.6 and a <= 0.8:
            description = 'bandreject'
        if a > 0.8:
            description = 'pitch'
        augmentation_chain = augmentation_factory(description, sampling_rate)
        y = augmentation_chain.apply(speech.cpu().float(),
        src_info=dict(rate=sampling_rate, length=speech.size(1), channels=speech.size(0)),
        target_info=dict(rate=sampling_rate, length=0)
        )
        if sample_room['net_input']['source'].shape[1] != y.shape[1]:
            return sample_room
        sample_room['net_input']['source'] = y.type_as(sample_room['net_input']['source'])
    return sample_room

def wapat_generator(sample, model, optimizer,AMPOptimizer, attack_type='Linf', eps=0.01, attack_steps=1, attack_lr=0.01, random_start_prob=0.0, targeted=False, attack_criterion='regular', use_best=True, eval_mode=True):

    target = True
    prev_training = bool(model.training)

    if eval_mode:
        model.eval()
    sample['net_input']["feature_em"] = None
    sample_room = copy.deepcopy(sample)
    with torch.no_grad():
        sample_room = sample_aug(sample_room)
   
    with torch.no_grad():
        adv_losses_res, sample_size, logging_output, feature_em, _ = attack_criterion(model, sample)
        del sample_size, logging_output 
    assert attack_type in ['Linf', 'L2'], '{} is not supported!'.format(attack_type)

    images = sample
    images['id'] = images['id'].detach()
    images['net_input']['source'] = images['net_input']['source'].detach()
    images['net_input']['padding_mask'] = images['net_input']['padding_mask'].detach()
    images['net_input']["feature_em"] = feature_em.detach()

    if target:
        n = torch.randperm(sample['target'].shape[0])
        images['target'] = images['target'][n]
        images['target_lengths'] =  images['target_lengths'][n]
        sample_room['target'] = images['target']
        sample_room['target_lengths'] = images['target_lengths']
    m = -1 if targeted else 1


    adv_losses_res_room, sample_size_room, logging_output_room, feature_em_room, res_lprobs_room = attack_criterion(model, sample_room)
    
    res_lprobs_room = copy.deepcopy(res_lprobs_room.detach())
    torch.mean(m * adv_losses_res_room).backward()
    grad_room = feature_em_room.grad.detach()
    feature_em_room = feature_em_room.detach()

    orig_input = feature_em.detach()
    if attack_type == 'Linf':
        step = LinfStep(eps=eps, orig_input=orig_input, step_size=attack_lr)
    elif attack_type == 'L2':
        step = L2Step(eps=eps, orig_input=orig_input, step_size=attack_lr)

    m = -1 if targeted else 1
    best_loss = None
    best_x = None

    if random.random() < random_start_prob:
        images = step.random_perturb(images)

    for _ in range(attack_steps):
        images['net_input']["feature_em"] = feature_em.clone().detach().requires_grad_(True)

        adv_losses, sample_size, logging_output, feature_em_, res_lprobs_ = attack_criterion(model, images)
        if feature_em_.shape != feature_em_room.shape:
            rir_loss = 0
        else:
            rir_loss = -1 * torch.cosine_similarity(feature_em_, feature_em_room).mean()
        torch.mean(m * (adv_losses+rir_loss)).backward(retain_graph=True)
        grad = images['net_input']['feature_em'].grad.detach()
        grad_new = (grad + grad_room) / 2
        
        optimizer.zero_grad()
        with torch.no_grad():
            varlist = [adv_losses, best_loss, feature_em, best_x, m]
            best_loss, best_x = replace_best(*varlist) if use_best else (adv_losses, feature_em)
            feature_em = step.step(feature_em, grad_new)
            feature_em = step.project(feature_em)

    adv_losses, sample_size, logging_output, feature_em_, _ = attack_criterion(model, images)
    varlist = [adv_losses, best_loss, feature_em, best_x, m]
    best_loss, best_x = replace_best(*varlist) if use_best else (adv_losses, feature_em)
    if prev_training:
        model.train() 
    return best_x, adv_losses_res, res_lprobs_room, feature_em_room
    
