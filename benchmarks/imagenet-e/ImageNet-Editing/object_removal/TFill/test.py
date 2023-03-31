import os,time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from options.test_options import TestOptions
from dataloader.data_loader import dataloader
from model import create_model
from itertools import islice
from util.visualizer import save_images
from util import html

if __name__=='__main__':
    opt = TestOptions().parse()       # get test options
    opt.name = 'imagenet'
    opt.img_file='../../tmp/img/'
    opt.mask_file='../../tmp/mask/'
    opt.results_dir='../../results'
    opt.model='tc'
    opt.coarse_or_refine='refine'
    opt.gpu_id=0
    opt.no_shuffle=True
    opt.batch_size=1
    opt.preprocess='scale_shortside'
    opt.mask_type=3
    opt.load_size=512
    opt.attn_G=True
    opt.add_noise=True
    dataset = dataloader(opt)       # create a dataset
    dataset_size = len(dataset) * opt.batch_size
    print('testing images = %d' % dataset_size)
    model = create_model(opt)       # create a model
    # create a website
    opt.epoch = '%d' % opt.which_iter if opt.which_iter > 0 else opt.epoch
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    opt.save_dir = web_dir
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    opt.how_many = dataset_size if opt.how_many == float("inf") else opt.how_many

    iter_data_time = time.time()
    for i, data in enumerate(islice(dataset, opt.how_many)):
        if i == 0:
            model.setup(opt)
            model.parallelize()
            model.eval()
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
    total_time = time.time() - iter_data_time
    print('the total evaluation time %f' % (total_time))
    webpage.save()
