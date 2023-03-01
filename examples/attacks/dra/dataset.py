import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from scipy.misc import imread, imresize, imsave


class CustomDataSet(Dataset):
    def __init__(self, input_dir, input_height, input_width):
        self.input_dir = input_dir
        self.input_size = [input_height, input_width]
        self.image_list = os.listdir(input_dir)

    def __getitem__(self, item):
        img_path = self.image_list[item]
        with open(os.path.join(self.input_dir, img_path), 'rb') as f:
            image = imresize(imread(f, mode='RGB'), self.input_size).transpose((2, 0, 1)).astype(np.float32) / 255.0
        return image, item

    def __len__(self):
        return len(self.image_list)


def load_images(input_dir, batch_size, input_height=224, input_width=224):
    """Read png images from input directory in batches.
        Args:
            input_dir: input directory
            batch_size: size of minibatch
            input_height: the array size of input
            input_width: the array size of input
        Return:
            dataloader
    """
    img_set = CustomDataSet(input_dir=input_dir, input_height=input_height, input_width=input_width)
    img_loader = DataLoader(img_set, batch_size=batch_size, num_workers=2)
    return img_loader, img_set.image_list


def save_images(images, img_list, idx, output_dir):
    """Saves images to the output directory.
        Args:
          images: tensor with minibatch of images
          img_list: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
          output_dir: directory where to save images
    """
    for i, sample_idx in enumerate(idx.numpy()):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        filename = img_list[sample_idx]
        cur_images = (images[i, :, :, :] * 255).astype(np.uint8)
        with open(os.path.join(output_dir, filename), 'wb') as f:
            imsave(f, cur_images.transpose(1, 2, 0), format='png')


if __name__ == '__main__':
    cdataset = CustomDataSet('nat_images', input_height=299, input_width=299)
    img, _ = cdataset.__getitem__(0)
    print(img.shape)
