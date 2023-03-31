import argparse
import os
import torch
import model
from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self, parser):
        # base define
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment.')
        parser.add_argument('--model', type=str, default='tc', help='name of the model type. [pluralistic]')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are save here')
        parser.add_argument('--which_iter', type=int, default='0', help='which iterations to load')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
        # data define
        parser.add_argument('--mask_type', type=int, default=[0,1,3], help='0:center,1:regular,2:irregular,3:external')
        parser.add_argument('--img_file', type=str, default='/data/dataset/train', help='training and testing dataset')
        parser.add_argument('--mask_file', type=str, default='none', help='load test mask')
        parser.add_argument('--img_nc', type=int, default=3, help='# of image channels')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='preprocessing image at load time')
        parser.add_argument('--load_size', type=int, default=542, help='scale examples to this size')
        parser.add_argument('--fine_size', type=int, default=512, help='then crop to this size')
        parser.add_argument('--fixed_size', type=int, default=256, help='fixed the image size in S1 with transformer')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the image')
        parser.add_argument('--data_powers', type=int, default=5, help='# times of the scale to 2 times')
        parser.add_argument('--reverse_mask', action='store_true', help='if specified, random reverse the mask region')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--nThreads', type=int, default=8, help='# threads for loading data')
        parser.add_argument('--no_shuffle', action='store_true', help='if true, takes examples serial')
        # display parameter define
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        parser.add_argument('--display_id', type=int, default=None, help='display id of the web')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='display name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8092, help='port of the web display')
        parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all examples in a single visidom web panel')
        # Encoder-Decoder define
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=32, help='# of dis filters in the first conv layer')
        parser.add_argument('--num_res_blocks', type=int, default=2, help='# of residual block in the encoder and decoder layer')
        parser.add_argument('--netD', type=str, default='style', help='specify discriminator architecture ')
        parser.add_argument('--netG', type=str, default='diff', help='specify decoder architecture')
        parser.add_argument('--netE', type=str, default='diff', help='specify encoder architecture')
        parser.add_argument('--kernel_G', type=int, default=3, help='kernel size for the decoder')
        parser.add_argument('--kernel_E', type=int, default=1, help='kernel size for the encoder')
        parser.add_argument('--add_noise', action='store_true', help='if true, add noise to the decoder')
        parser.add_argument('--attn_E', action='store_true', help='if true, use attention in the encoder')
        parser.add_argument('--attn_G', action='store_true', help='if true, use attention in the decoder')
        parser.add_argument('--attn_D', action='store_true', help='if true, use attention in the decoder')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--n_layers_G', type=int, default=4, help='# of down sample layers in the Encoder and Decoder')
        parser.add_argument('--norm', type=str, default='pixel', help='instance normalization or batch normalization [instance | batch | pixel | none]')
        parser.add_argument('--activation', type=str, default='leakyrelu', help='activation layer [relu | gelu | leakyrelu | none]')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--lipip_path', type=str, default='./model/lpips/vgg.pth', help='the pretrained LIPPS model')
        # Transformer define
        parser.add_argument('--netT', type=str, default='original', help='specify transformer architecture')
        parser.add_argument('--embed_dim', type=int, default=512, help='the numbers of embedding dimension')
        parser.add_argument('--dropout', type=float, default=0., help='the dropout probability in transformer')
        parser.add_argument('--kernel_T', type=int, default=1, help='kernel size for the transformer projection')
        parser.add_argument('--n_encoders', type=int, default=12, help='the numbers of encoder in transformer')
        parser.add_argument('--n_decoders', type=int, default=0, help='the numbers of decoder in transformer')
        parser.add_argument('--embed_type', type=str, default='learned', choices=['learned', 'sine'])
        parser.add_argument('--top_k', type=int, default=10, help='sample the results on top k value')
        # VQ define
        parser.add_argument('--num_embeds', type=int, default=1024, help='the numbers of words for image')
        parser.add_argument('--use_pos_G', action='store_true', help='if true, position embedding in G')
        parser.add_argument('--word_size', type=int, default=16, help='the numbers of word for each image')
        self.initialized = True
        return parser

    def gather_options(self):
        """Add additional model-specific options"""
        if not self.initialized:
            parser = self.initialize(self.parser)

        # get basic options
        opt, _ = parser.parse_known_args()

        # modify the options for different models
        model_option_set = model.get_option_setter(opt.model)
        parser = model_option_set(parser, self.isTrain)
        opt = parser.parse_args()

        return opt

    def parse(self):
        """Parse the options"""
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids):
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt

        return self.opt

    @staticmethod
    def print_options(opt):
        """print and save options"""
        print('--------------Options--------------')
        for k, v in sorted(vars(opt).items()):
            print('%s: %s' % (str(k), str(v)))
        print('----------------End----------------')

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        if opt.isTrain:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('--------------Options--------------\n')
            for k, v in sorted(vars(opt).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('----------------End----------------\n')