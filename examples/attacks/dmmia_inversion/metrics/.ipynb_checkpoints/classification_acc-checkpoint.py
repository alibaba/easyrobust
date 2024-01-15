import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from utils.stylegan import create_image

from metrics.accuracy import Accuracy, AccuracyTopK


class ClassificationAccuracy():
    def __init__(self, evaluation_network, device='cuda:0'):
        self.evaluation_network = evaluation_network
        self.device = device

    def compute_acc(self, w, targets, generator, config, batch_size=64, resize=299, rtpt=None, final_imgs=None):
        self.evaluation_network.eval()
        self.evaluation_network.to(self.device)
        if final_imgs is not None:
            dataset = TensorDataset(final_imgs, targets)
        else:
            dataset = TensorDataset(w, targets)
        acc_top1 = Accuracy()
        acc_top5 = AccuracyTopK(k=5)
        predictions = []
        correct_confidences = []
        total_confidences = []
        maximum_confidences = []

        max_iter = math.ceil(len(dataset) / batch_size)

        with torch.no_grad():
            for step, (w_batch, target_batch) in enumerate(DataLoader(dataset,
                                                                      batch_size=batch_size,
                                                                      shuffle=False)):
                if final_imgs is not None:
                    imgs = w_batch.to(self.device)
                    target_batch = target_batch.to(self.device)
                else:
                    w_batch, target_batch = w_batch.to(
                        self.device), target_batch.to(self.device)
                    imgs = create_image(
                        w_batch, generator, config.attack_center_crop, resize=resize, batch_size=batch_size, device = self.device)
                imgs = imgs.to(self.device)
                output = self.evaluation_network(imgs)

                acc_top1.update(output, target_batch)
                acc_top5.update(output, target_batch)

                pred = torch.argmax(output, dim=1)
                predictions.append(pred)
                confidences = output.softmax(1)
                target_confidences = torch.gather(confidences, 1,
                                                  target_batch.unsqueeze(1))
                correct_confidences.append(
                    target_confidences[pred == target_batch])
                total_confidences.append(target_confidences)
                maximum_confidences.append(torch.max(confidences, dim=1)[0])

            acc_top1 = acc_top1.compute_metric()
            acc_top5 = acc_top5.compute_metric()
            correct_confidences = torch.cat(correct_confidences,
                                            dim=0)
            avg_correct_conf = correct_confidences.mean().cpu().item()
            confidences = torch.cat(total_confidences, dim=0).cpu()
            confidences = torch.flatten(confidences)
            maximum_confidences = torch.cat(maximum_confidences,
                                            dim=0).cpu().tolist()
            avg_total_conf = torch.cat(total_confidences,
                                       dim=0).mean().cpu().item()
            predictions = torch.cat(predictions, dim=0).cpu()

            # Compute class-wise precision
            target_list = targets.cpu().tolist()
            precision_list = [['target', 'mean_conf', 'precision']]
            targets = targets.to('cuda:0')
            for t in set(target_list):
                mask = torch.where(targets == t, True, False)
                conf_masked = confidences[mask]
                precision = torch.sum(
                    predictions[mask] == t) / torch.sum(targets == t)
                precision_list.append(
                    [t, conf_masked.mean().item(), precision.cpu().item()])
            confidences = confidences.tolist()
            predictions = predictions.tolist()

            if rtpt:
                rtpt.step(
                    subtitle=f'Classification Evaluation step {step} of {max_iter}')

        return acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, \
            confidences, maximum_confidences, precision_list
