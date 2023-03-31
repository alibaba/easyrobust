from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test examples to run')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--nsampling', type=int, default=1, help='ramplimg # times for each examples')

        self.isTrain = False

        return parser