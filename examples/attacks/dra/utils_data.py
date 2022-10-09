import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class SubsetImageNet(Dataset):
    def __init__(self, root, class_to_idx='./imagenet_class_to_idx.npy', transform=None):
        super(SubsetImageNet, self).__init__()
        self.root = root
        self.transform = transform
        img_path = os.listdir(root)
        img_path = sorted(img_path)
        self.img_path = [item for item in img_path if 'png' in item]
        self.class_to_idx = np.load(class_to_idx, allow_pickle=True)[()]

    def __getitem__(self, item):
        filepath = os.path.join(self.root, self.img_path[item])
        sample = Image.open(filepath, mode='r')

        if self.transform:
            sample = self.transform(sample)

        class_name = self.img_path[item].split('_')[0]
        label = self.class_to_idx[class_name]

        return sample, label, item

    def __len__(self):
        return len(self.img_path)


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
        cur_images = (images[i, :, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)

        im = Image.fromarray(cur_images)
        im.save('{}.png'.format(os.path.join(output_dir, filename)))
