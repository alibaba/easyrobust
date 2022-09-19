import os
from PIL import Image
import torch
from torchvision.datasets import ImageFolder

from easyrobust.parallel import is_main_process

class StylizedImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, split, transform=None, use_clean=True):
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        self._indices = []
        self.use_clean = use_clean

        if self.split == 'train':
            sin_base_dir = os.path.join(self.data_dir, 'Stylized')
            if not os.path.exists(sin_base_dir):
                raise Exception('Stylized ImageNet data is not found! please refer https://github.com/rgeirhos/Stylized-ImageNet to generate first')

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
                if self.use_clean:
                    self._indices.append((os.path.join(self.data_dir, img_path), label))
                self._indices.append((os.path.join(sin_base_dir, img_path.split('/')[1], img_path.split('/')[2].replace('.JPEG', '.png')), label))

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
