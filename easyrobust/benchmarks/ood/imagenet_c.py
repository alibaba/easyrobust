import os
import torch
from tqdm import tqdm

from torchvision import transforms, datasets
from timm.utils import AverageMeter, accuracy

def get_mce_from_accuracy(accuracy, error_alexnet):
    """Computes mean Corruption Error from accuracy"""
    error = 100. - accuracy
    ce = error / (error_alexnet * 100.)

    return ce

def get_ce_alexnet():
    """Returns Corruption Error values for AlexNet"""
    ce_alexnet = dict()
    ce_alexnet['Gaussian Noise'] = 0.886428
    ce_alexnet['Shot Noise'] = 0.894468
    ce_alexnet['Impulse Noise'] = 0.922640
    ce_alexnet['Defocus Blur'] = 0.819880
    ce_alexnet['Glass Blur'] = 0.826268
    ce_alexnet['Motion Blur'] = 0.785948
    ce_alexnet['Zoom Blur'] = 0.798360
    ce_alexnet['Snow'] = 0.866816
    ce_alexnet['Frost'] = 0.826572
    ce_alexnet['Fog'] = 0.819324
    ce_alexnet['Brightness'] = 0.564592
    ce_alexnet['Contrast'] = 0.853204
    ce_alexnet['Elastic Transform'] = 0.646056
    ce_alexnet['Pixelate'] = 0.717840
    ce_alexnet['JPEG Compression'] = 0.606500

    return ce_alexnet

data_loaders_names = {
        'Brightness': 'brightness',
        'Contrast': 'contrast',
        'Defocus Blur': 'defocus_blur',
        'Elastic Transform': 'elastic_transform',
        'Fog': 'fog',
        'Frost': 'frost',
        'Gaussian Noise': 'gaussian_noise',
        'Glass Blur': 'glass_blur',
        'Impulse Noise': 'impulse_noise',
        'JPEG Compression': 'jpeg_compression',
        'Motion Blur': 'motion_blur',
        'Pixelate': 'pixelate',
        'Shot Noise': 'shot_noise',
        'Snow': 'snow',
        'Zoom Blur': 'zoom_blur'
    }

def evaluate_imagenet_c(model, data_dir, test_batchsize=128, test_transform=None):
    if not os.path.exists(data_dir):
        print('{} is not exist. skip')
        return
    
    device = next(model.parameters()).device
    result_dict = {}
    ce_alexnet = get_ce_alexnet()

    # imagenet-c always has size of 224
    if test_transform is None:
        inc_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        inc_transform = test_transform

    for name, subdir in data_loaders_names.items():
        print('run {}...'.format(name))
        for severity in range(1, 6):
            inc_dataset = datasets.ImageFolder(os.path.join(data_dir, subdir, str(severity)), transform=inc_transform)
            inc_data_loader = torch.utils.data.DataLoader(
                            inc_dataset, sampler=None,
                            batch_size=test_batchsize,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False
                        )
            
            top1_m = AverageMeter()
            model.eval()
            for input, target in tqdm(inc_data_loader):
                input = input.to(device)
                target = target.to(device)
                with torch.no_grad():
                    output = model(input)
                acc1, _ = accuracy(output, target, topk=(1, 5))
                top1_m.update(acc1.item(), output.size(0))

            # print(f"Accuracy on the {name+'({})'.format(severity)}: {top1_m.avg:.1f}%")
            result_dict[name+'({})'.format(severity)] = top1_m.avg

    mCE = 0
    counter = 0
    overall_acc = 0
    for name, _ in data_loaders_names.items():
        acc_top1 = 0
        for severity in range(1, 6):
            acc_top1 += result_dict[name+'({})'.format(severity)]
        acc_top1 /= 5
        CE = get_mce_from_accuracy(acc_top1, ce_alexnet[name])
        mCE += CE
        overall_acc += acc_top1
        counter += 1
        print("{0}: Top1 accuracy {1:.2f}, CE: {2:.2f}".format(
                name, acc_top1, 100. * CE))
    
    overall_acc /= counter
    mCE /= counter
    print("Top1 accuracy {0:.1f}%, mCE: {1:.1f} on the ImageNet-C".format(overall_acc, mCE * 100.))
