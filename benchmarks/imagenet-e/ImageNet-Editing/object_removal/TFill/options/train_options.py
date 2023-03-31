from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # training parameters
        parser.add_argument('--iter_count', type=int, default=0, help='the starting epoch count')
        parser.add_argument('--n_iter', type=int, default=20000000, help='# of iter with initial learning rate')
        parser.add_argument('--n_iter_decay', type=int, default=00000000, help='# of iter to decay learning rate to zero')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        # learning rate and loss weight
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy[lambda|step|plateau]')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--gan_mode', type=str, default='nonsaturating', choices=['hinge', 'lsgan', 'standard', 'wgan-gp', 'nonsaturating'])
        # display the results
        parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=3, help='if positive, display all examples in a single visdom web panel with certain number of examples per row.')
        parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_iters_freq', type=int, default=100000, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results')

        self.isTrain = True

        return parser