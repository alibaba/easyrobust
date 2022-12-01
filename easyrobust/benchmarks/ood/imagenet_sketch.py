import os
import torch
from tqdm import tqdm

from torchvision import transforms, datasets
from timm.utils import AverageMeter, reduce_tensor, accuracy

def evaluate_imagenet_sketch(model, data_dir, test_batchsize=128, test_transform=None, dist=False):
    if not os.path.exists(data_dir):
        print('{} is not exist. skip')
        return

    if dist:
        assert torch.distributed.is_available() and torch.distributed.is_initialized()
    
    device = next(model.parameters()).device

    if test_transform is None:
        insk_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        insk_transform = test_transform

    dataset_insk = datasets.ImageFolder(data_dir, transform=insk_transform)

    sampler = None
    if dist:
        sampler = torch.utils.data.DistributedSampler(dataset_insk, shuffle=False)


    insk_data_loader = torch.utils.data.DataLoader(
                    dataset_insk, sampler=sampler,
                    batch_size=test_batchsize,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=False
                )
            
    top1_m = AverageMeter()
    model.eval()
    for input, target in tqdm(insk_data_loader):
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(input)

        acc1, _ = accuracy(output, target, topk=(1, 5))
        if dist:
            acc1 = reduce_tensor(acc1, torch.distributed.get_world_size())
            torch.cuda.synchronize()

        top1_m.update(acc1.item(), output.size(0))

    print(f"Top1 Accuracy on the ImageNet-Sketch: {top1_m.avg:.1f}%")
    return top1_m.avg
