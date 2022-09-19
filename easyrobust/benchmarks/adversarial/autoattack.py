import os
import json
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from timm.utils import AverageMeter, accuracy
from easyrobust.attacks import AutoAttack

class AutoAttackImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self._indices = []
        for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_test_image_ids.txt')):
            img_path = line.strip()
            class_map = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_class_to_id_map.json')))
            class_ids, _ = img_path.split('/')
            self._indices.append((img_path, class_map[class_ids]))

    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index]
        img = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')
        label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def evaluate_imagenet_autoattack(model, data_dir, test_batchsize=128, test_transform=None, attack_type='Linf', epsilon=4/255):
    if not os.path.exists(data_dir):
        print('{} is not exist. skip')
        return

    device = next(model.parameters()).device
    assert attack_type in ['Linf', 'L2'], '{} is not supported!'.format(attack_type)
    adversary = AutoAttack(model, norm=attack_type, eps=epsilon, version='standard')

    if test_transform is None:
        in_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        in_transform = test_transform

    print(test_transform)

    dataset_in = AutoAttackImageNetDataset(data_dir, transform=in_transform)

    in_data_loader = torch.utils.data.DataLoader(
                    dataset_in, sampler=None,
                    batch_size=test_batchsize,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=False
                )
            
    top1_m = AverageMeter()
    model.eval()
    for input, target in tqdm(in_data_loader):
        input = input.to(device)
        target = target.to(device)

        x_adv = adversary.run_standard_evaluation(input, target, bs=input.shape[0])
        with torch.no_grad():
            labels_adv = model(x_adv.detach())

        acc1, _ = accuracy(labels_adv, target, topk=(1, 5))
        top1_m.update(acc1.item(), labels_adv.size(0))

    print(f"Top1 Accuracy on the AutoAttack: {top1_m.avg:.1f}%")


