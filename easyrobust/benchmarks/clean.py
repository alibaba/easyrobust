import os
import torch
from tqdm import tqdm

from torchvision import transforms, datasets
from timm.utils import AverageMeter, accuracy

def evaluate_imagenet_val(model, data_dir, test_batchsize=128, test_transform=None):
    if not os.path.exists(data_dir):
        print('{} is not exist. skip')
        return
    
    device = next(model.parameters()).device

    if test_transform is None:
        in_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        in_transform = test_transform

    dataset_in = datasets.ImageFolder(data_dir, transform=in_transform)
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
        with torch.no_grad():
            output = model(input)

        acc1, _ = accuracy(output, target, topk=(1, 5))
        top1_m.update(acc1.item(), output.size(0))

    print(f"Top1 Accuracy on the ImageNet-Val: {top1_m.avg:.1f}%")
