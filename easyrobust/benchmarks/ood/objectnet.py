import os
import torch
from tqdm import tqdm

from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from timm.utils import AverageMeter, reduce_tensor, accuracy

objectnet_mapping = {"409": 1, "530": 1, "414": 2, "954": 4, "419": 5, "790": 8, "434": 9, "440": 13, "703": 16, "671": 17, "444": 17, "446": 20, "455": 29, "930": 35, "462": 38, "463": 39, "499": 40, "473": 45, "470": 46, "487": 48, "423": 52, "559": 52, "765": 52, "588": 57, "550": 64, "507": 67, "673": 68, "846": 75, "533": 78, "539": 81, "630": 86, "740": 88, "968": 89, "729": 92, "549": 98, "545": 102, "567": 109, "578": 83, "589": 112, "587": 115, "560": 120, "518": 120, "606": 124, "608": 128, "508": 131, "618": 132, "619": 133, "620": 134, "951": 138, "623": 139, "626": 142, "629": 143, "644": 149, "647": 150, "651": 151, "659": 153, "664": 154, "504": 157, "677": 159, "679": 164, "950": 171, "695": 173, "696": 175, "700": 179, "418": 182, "749": 182, "563": 182, "720": 188, "721": 190, "725": 191, "728": 193, "923": 196, "731": 199, "737": 200, "811": 201, "742": 205, "761": 210, "769": 216, "770": 217, "772": 218, "773": 219, "774": 220, "783": 223, "792": 229, "601": 231, "655": 231, "689": 231, "797": 232, "804": 235, "806": 236, "809": 237, "813": 238, "632": 239, "732": 248, "759": 248, "828": 250, "850": 251, "834": 253, "837": 255, "841": 256, "842": 257, "610": 258, "851": 259, "849": 268, "752": 269, "457": 273, "906": 273, "859": 275, "999": 276, "412": 284, "868": 286, "879": 289, "882": 292, "883": 293, "893": 297, "531": 298, "898": 299, "543": 302, "778": 303, "479": 304, "694": 304, "902": 306, "907": 307, "658": 309, "909": 310}
objectnet_mapping = {int(k): v for k, v in objectnet_mapping.items()}

def imageNetIDToObjectNetID(prediction_class):
    for i in range(len(prediction_class)):
        if prediction_class[i] in objectnet_mapping:
            prediction_class[i] = objectnet_mapping[prediction_class[i]]
        else:
            prediction_class[i] = -1

# a dataset define which returns img_path
class ObjectNetDataset(datasets.DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ObjectNetDataset, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = []
        label_set = set()
        for k, v in objectnet_mapping.items():
            label_set.add(v)

        # filter samples non-overlap with imagenet
        for img_path, label in self.samples:
            if label in label_set:
                self.imgs.append((img_path, label))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        path, target = self.imgs[index]
        sample = self.loader(path)
        width, height = sample.size
        cropArea = (2, 2, width-2, height-2)
        sample = sample.crop(cropArea)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def evaluate_objectnet(model, data_dir, test_batchsize=128, test_transform=None, dist=False):
    if not os.path.exists(data_dir):
        print('{} is not exist. skip')
        return

    if dist:
        assert torch.distributed.is_available() and torch.distributed.is_initialized()

    device = next(model.parameters()).device

    if test_transform is None:
        objectnet_transform = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        objectnet_transform = test_transform

    dataset_objnet = ObjectNetDataset(data_dir, transform=objectnet_transform)

    sampler = None
    if dist:
        sampler = torch.utils.data.DistributedSampler(dataset_objnet, shuffle=False)

    objnet_data_loader = torch.utils.data.DataLoader(
                    dataset_objnet, sampler=sampler,
                    batch_size=test_batchsize,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=False
                )
            
    top1_m = AverageMeter()
    model.eval()
    for input, target in tqdm(objnet_data_loader):
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(input)

        _, prediction_class = output.topk(5, 1, True, True)
        prediction_class = prediction_class.data.cpu().tolist()
        for i in range(output.size(0)):
            imageNetIDToObjectNetID(prediction_class[i])

        prediction_class = torch.tensor(prediction_class).to(device)
        prediction_class = prediction_class.t()
        correct = prediction_class.eq(target.reshape(1, -1).expand_as(prediction_class))
        acc1, _ = [correct[:k].reshape(-1).float().sum(0) * 100. / output.size(0) for k in (1, 5)]
        if dist:
            acc1 = reduce_tensor(acc1, torch.distributed.get_world_size())
            torch.cuda.synchronize()

        top1_m.update(acc1.item(), output.size(0))

    print(f"Top1 Accuracy on the ObjectNet: {top1_m.avg:.1f}%")
    return top1_m.avg