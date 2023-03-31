import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks, losses


class C(BaseModel):
    """This class implements the conv-based model for image completion"""
    def name(self):
        return "Conv-based Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--coarse_or_refine', type=str, default='coarse', help='train the transform or refined network')
        parser.add_argument('--down_layers', type=int, default=4, help='# times down sampling for refine generator')
        if is_train:
            parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for discriminator loss')
            parser.add_argument('--lambda_lp', type=float, default=10.0, help='weight for the perceptual loss')
            parser.add_argument('--lambda_gradient', type=float, default=0.0, help='weight for the gradient penalty')

        return parser

    def __init__(self, opt):
        """inital the Transformer model"""
        BaseModel.__init__(self, opt)
        self.visual_names = ['img', 'img_m', 'img_g', 'img_out']
        self.model_names = ['E', 'G', 'D',]
        self.loss_names = ['G_rec', 'G_lp', 'G_GAN', 'D_real', 'D_fake']

        self.netE = networks.define_E(opt)
        self.netG = networks.define_G(opt)
        self.netD = networks.define_D(opt, opt.fixed_size)

        if 'refine' in self.opt.coarse_or_refine:
            opt = self._refine_opt(opt)
            self.netG_Ref = networks.define_G(opt)
            self.netD_Ref = networks.define_D(opt, opt.fine_size)
            self.visual_names += ['img_ref', 'img_ref_out']
            self.model_names += ['G_Ref', 'D_Ref']

        if self.isTrain:
            # define the loss function
            self.L1loss = torch.nn.L1Loss()
            self.GANloss = losses.GANLoss(opt.gan_mode).to(self.device)
            self.NormalVGG = losses.Normalization(self.device)
            self.LPIPSloss = losses.LPIPSLoss(ckpt_path=opt.lipip_path).to(self.device)
            if len(self.opt.gpu_ids) > 0:
                self.LPIPSloss = torch.nn.parallel.DataParallel(self.LPIPSloss, self.opt.gpu_ids)
            # define the optimizer
            if 'coarse' in self.opt.coarse_or_refine:
                self.optimizerG = torch.optim.Adam(list(self.netE.parameters()) + list(self.netG.parameters()),
                                                   lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * 4, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizerG)
                self.optimizers.append(self.optimizerD)
            if 'refine' in self.opt.coarse_or_refine:
                self.optimizerGRef = torch.optim.Adam(self.netG_Ref.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizerDRef = torch.optim.Adam(self.netD_Ref.parameters(), lr=opt.lr * 4, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizerGRef)
                self.optimizers.append(self.optimizerDRef)
        else:
            self.visual_names = ['img', 'img_m']

    def set_input(self, input):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input

        self.image_paths = self.input['img_path']
        self.img_org = input['img_org'].to(self.device) * 2 - 1
        self.img = input['img'].to(self.device) * 2 - 1
        self.mask = input['mask'].to(self.device)

        # get I_m and I_c for image with mask and complement regions for training
        self.img_m = self.mask * self.img_org

    @torch.no_grad()
    def test(self):
        """Run forward processing for testing"""
        fixed_img = F.interpolate(self.img_m, size=self.img.size()[2:], mode='bicubic', align_corners=True).clamp(-1, 1)
        fixed_mask = (F.interpolate(self.mask, size=self.img.size()[2:], mode='bicubic', align_corners=True) > 0.9).type_as(fixed_img)
        out, mask = self.netE(fixed_img, mask=fixed_mask, return_mask=True)

        # sample result
        for i in range(self.opt.nsampling):
            img_g = self.netG(out)
            img_g_org = F.interpolate(img_g, size=self.img_org.size()[2:], mode='bicubic', align_corners=True).clamp(-1,1)
            img_out = self.mask * self.img_org + (1 - self.mask) * img_g_org
            self.save_results(img_out, path=self.opt.save_dir + '/img_out', data_name=i)
            if 'refine' in self.opt.coarse_or_refine:
                img_ref = self.netG_Ref(img_out, mask=self.mask)
                img_ref_out = self.mask * self.img_org + (1 - self.mask) * img_ref
                self.save_results(img_ref_out, path=self.opt.save_dir + '/img_ref_out', data_name=i)

    def forward(self):
        """Run forward processing to get the outputs"""
        fixed_img = F.interpolate(self.img_m, size=self.img.size()[2:], mode='bicubic', align_corners=True).clamp(-1, 1)
        self.fixed_mask = (F.interpolate(self.mask, size=self.img.size()[2:], mode='bicubic', align_corners=True) > 0.9).type_as(fixed_img)
        out, mask = self.netE(fixed_img, mask=self.fixed_mask, return_mask=True)
        self.img_g = self.netG(out)
        img_g_org = F.interpolate(self.img_g, size=self.img_org.size()[2:], mode='bicubic', align_corners=True).clamp(-1, 1)
        self.img_out = self.mask * self.img_org + (1 - self.mask) * img_g_org

        if 'refine' in self.opt.coarse_or_refine:
            self.img_ref = self.netG_Ref(self.img_out, self.mask)
            self.img_ref_out = self.mask * self.img_org + (1 - self.mask) * self.img_ref

    def backward_D_basic(self, netD, real, fake):
        """
        Calculate GAN loss for the discriminator
        :param netD: the discriminator D
        :param real: real examples
        :param fake: examples generated by a generator
        :return: discriminator loss
        """
        self.loss_D_real = self.GANloss(netD(real), True, is_dis=True)
        self.loss_D_fake = self.GANloss(netD(fake), False, is_dis=True)
        loss_D = self.loss_D_real + self.loss_D_fake
        if self.opt.lambda_gradient > 0:
            self.loss_D_Gradient, _ = losses.cal_gradient_penalty(netD, real, fake, real.device, lambda_gp=self.opt.lambda_gradient)
            loss_D += self.loss_D_Gradient
        loss_D.backward()
        return loss_D

    def backward_D(self):
        """Calculate the GAN loss for discriminator"""
        self.loss_D = 0
        if 'coarse' in self.opt.coarse_or_refine:
            self.set_requires_grad([self.netD], True)
            self.optimizerD.zero_grad()
            real = self.img.detach()
            fake = self.img_g.detach()
            self.loss_D += self.backward_D_basic(self.netD, real, fake) if self.opt.lambda_g > 0 else 0
        if 'refine' in self.opt.coarse_or_refine:
            self.set_requires_grad([self.netD_Ref], True)
            self.optimizerDRef.zero_grad()
            real = self.img_org.detach()
            fake = self.img_ref.detach()
            self.loss_D += self.backward_D_basic(self.netD_Ref, real, fake) if self.opt.lambda_g > 0 else 0

    def backward_G(self):
        """Calculate the loss for generator"""
        self.loss_G_GAN = 0
        self.loss_G_rec = 0
        self.loss_G_lp =0
        if 'coarse' in self.opt.coarse_or_refine:
            self.set_requires_grad([self.netD], False)
            self.optimizerG.zero_grad()
            self.loss_G_GAN += self.GANloss(self.netD(self.img_g), True) * self.opt.lambda_g if self.opt.lambda_g > 0 else 0
            self.loss_G_rec += (self.L1loss(self.img_g * (1 - self.fixed_mask), self.img * (1 - self.fixed_mask)) * 3 +
                                self.L1loss(self.img_g * self.fixed_mask, self.img_g * self.fixed_mask)) * self.opt.lambda_rec
            norm_real = self.NormalVGG((self.img + 1) * 0.5)
            norm_fake = self.NormalVGG((self.img_g + 1) * 0.5)
            self.loss_G_lp += (self.LPIPSloss(norm_real, norm_fake).mean()) * self.opt.lambda_lp if self.opt.lambda_lp > 0 else 0
        if 'refine' in self.opt.coarse_or_refine:
            self.set_requires_grad([self.netD_Ref], False)
            self.optimizerGRef.zero_grad()
            self.loss_G_GAN += self.GANloss(self.netD_Ref(self.img_ref), True) * self.opt.lambda_g if self.opt.lambda_g > 0 else 0
            self.loss_G_rec += (self.L1loss(self.img_ref * (1 - self.mask), self.img_org * (1 - self.mask)) * 3 +
                                self.L1loss(self.img_ref * self.mask, self.img_org * self.mask)) * self.opt.lambda_rec
            norm_real = self.NormalVGG((self.img_org + 1) * 0.5)
            norm_fake = self.NormalVGG((self.img_ref + 1) * 0.5)
            self.loss_G_lp += (self.LPIPSloss(norm_real, norm_fake).mean()) * self.opt.lambda_lp if self.opt.lambda_lp > 0 else 0

        self.loss_G = self.loss_G_GAN + self.loss_G_rec + self.loss_G_lp

        self.loss_G.backward()

    def optimize_parameters(self):
        """update network weights"""
        # forward
        self.set_requires_grad([self.netE, self.netG], 'coarse' in self.opt.coarse_or_refine)
        self.forward()
        # update D
        self.backward_D()
        if 'coarse' in self.opt.coarse_or_refine:
            self.optimizerD.step()
        if 'refine' in self.opt.coarse_or_refine:
            self.optimizerDRef.step()
        # update G
        self.backward_G()
        if 'coarse' in self.opt.coarse_or_refine:
            self.optimizerG.step()
        if 'refine' in self.opt.coarse_or_refine:
            self.optimizerGRef.step()

    def _refine_opt(self, opt):
        """modify the opt for refine generator and discriminator"""
        opt.netG = 'refine'
        opt.netD = 'style'
        opt.attn_D = True

        return opt