import os
from PIL import Image
import torch
from torchvision.datasets import ImageFolder

from easyrobust.parallel import is_main_process

class DeepAugmentDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, split, transform=None):
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        self._indices = []

        if self.split == 'train':
            cae_base_dir = os.path.join(self.data_dir, 'CAE')
            edsr_base_dir = os.path.join(self.data_dir, 'EDSR')
            if not os.path.exists(cae_base_dir) or not os.path.exists(edsr_base_dir):
                raise Exception('deepaugment data is not found! please refer https://github.com/hendrycks/imagenet-r/tree/master/DeepAugment to generate first')

            if not os.path.exists(os.path.join(self.data_dir, 'train_with_label.txt')):
                if is_main_process():
                    print('train_with_label.txt not found, generate...')
                    temp_dset = ImageFolder(os.path.join(self.data_dir, 'train'))
                    with open(os.path.join(self.data_dir, 'train_with_label.txt'), 'w') as f:
                        for path, label in temp_dset.imgs:
                            path = '/'.join(path.split('/')[-3:])
                            f.write(path+'\t'+str(label)+'\n')

            for line in open(os.path.join(self.data_dir, 'train_with_label.txt'), encoding="utf-8"):
                img_path, label = line.strip().split('\t')
                self._indices.append((os.path.join(self.data_dir, img_path), label))
                # add deepaugment data
                self._indices.append((os.path.join(cae_base_dir, img_path.split('/')[1], img_path.split('/')[2]), label))
                self._indices.append((os.path.join(edsr_base_dir, img_path.split('/')[1], img_path.split('/')[2]), label))

        elif self.split == 'validation':
            if not os.path.exists(os.path.join(self.data_dir, 'val_with_label.txt')):
                if is_main_process():
                    print('val_with_label.txt not found, generate...')
                    temp_dset = ImageFolder(os.path.join(self.data_dir, 'val'))
                    with open(os.path.join(self.data_dir, 'val_with_label.txt'), 'w') as f:
                        for path, label in temp_dset.imgs:
                            path = '/'.join(path.split('/')[-3:])
                            f.write(path+'\t'+str(label)+'\n')

            for line in open(os.path.join(self.data_dir, 'val_with_label.txt'), encoding="utf-8"):
                img_path, label = line.strip().split('\t')
                self._indices.append((os.path.join(self.data_dir, img_path), label))

        else:
            raise Exception('split must be specified as train or validation!')

    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
