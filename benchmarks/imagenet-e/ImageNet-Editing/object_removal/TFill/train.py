import time
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    dataset = dataloader(opt)  # create a dataset
    dataset_size = len(dataset) * opt.batch_size
    print('training images = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    visualizer = Visualizer(opt)  # create a visualizer

    total_iters = opt.iter_count  # the total number of training iterations
    epoch = 0
    max_iteration = opt.n_iter + opt.n_iter_decay

    while (total_iters < max_iteration):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch += 1  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0
        visualizer.reset()  # reset the visualizer

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            if total_iters == 0:
                model.setup(opt)
                model.parallelize()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.log_imgs()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                model.save_networks('latest')

            if total_iters % opt.save_iters_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of iters %d' % (total_iters))
                model.save_networks('latest')
                model.save_networks(total_iters)

        print('End of iters %d / %d \t Time Taken: %d sec' % (total_iters, max_iteration, time.time() - epoch_start_time))
        model.update_learning_rate()