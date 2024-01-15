import torch

from metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, name='Accuracy'):
        super().__init__(name)

    def compute_metric(self):
        accuracy = self._num_corrects / self._num_samples
        return accuracy


class AccuracyTopK(BaseMetric):
    def __init__(self, name='accuracy_5', k=5):
        self.k = k
        super().__init__(name)

    def update(self, model_output, y_true):
        y_pred = torch.topk(model_output, dim=1, k=self.k).indices
        num_corrects = 0
        for k in range(self.k):
            num_corrects += torch.sum(y_pred[:, k] == y_true).item()
        self._num_corrects += num_corrects
        self._num_samples += y_true.shape[0]

    def compute_metric(self):
        accuracy = self._num_corrects / self._num_samples
        return accuracy
