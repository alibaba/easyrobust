import numpy as np
import torch
from torchvision.transforms import ToPILImage

from .base import AbstractModel


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def undo_default_preprocessing(images):
    """Convenience function: undo standard preprocessing."""

    assert type(images) is torch.Tensor
    default_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device())
    default_std = torch.Tensor([0.229, 0.224, 0.225]).to(device())

    images *= default_std[None, :, None, None]
    images += default_mean[None, :, None, None]

    return images


class PytorchModel(AbstractModel):

    def __init__(self, model, model_name, *args):
        self.model = model
        self.model_name = model_name
        self.args = args
        self.model.to(device())

    def to_numpy(self, x):
        if x.is_cuda:
            return x.detach().cpu().numpy()
        else:
            return x.numpy()

    def softmax(self, logits):
        assert type(logits) is np.ndarray

        softmax_op = torch.nn.Softmax(dim=1)
        softmax_output = softmax_op(torch.Tensor(logits))
        return self.to_numpy(softmax_output)

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        self.model.eval()
        logits = self.model(images)
        return self.to_numpy(logits)


class TimmModel(PytorchModel):

    def __init__(self, model, model_name, test_transform=None, *args):
        self.test_transform = test_transform
        super(TimmModel, self).__init__(model, model_name, args)

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()
        if self.test_transform:
            images = undo_default_preprocessing(images)
            images = [self.test_transform(ToPILImage()(image)) for image in images]
            images = torch.Tensor(np.stack(images, axis=0)).to(device())

        logits = self.model(images)
        return self.to_numpy(logits)
