from autoattack import AutoAttack
from models import *
from utils import *
import argparse
import sys
import os
sys.path.insert(0, '..')

from utils02 import (upper_limit, lower_limit, std, clamp, get_loaders, ImageNet_get_loaders,New_ImageNet_get_loaders_64,
    evaluate_pgd, evaluate_standard)
m_sample_list = []
all_block_count = 0
block_count = 0
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self,prob, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        identity = x.clone()
        global m_sample_list
        global all_block_count
        if self.training:
            if len(m_sample_list) != 8:
                all_block_count = all_block_count + 1
                m_sample = self.m.sample()
                m_sample_list.append(m_sample)
                if torch.equal(m_sample, torch.ones(1)):
                    print("******************")
                    self.conv1.weight.requires_grad = True
                    self.conv2.weight.requires_grad = True
                    out = F.relu(self.bn1(x))
                    shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
                    out = self.conv1(out)
                    out = self.conv2(F.relu(self.bn2(out)))
                    out += shortcut
                else:
                    print("!!!!!!!!!!!!!!!!!!")
                    self.conv1.weight.requires_grad = False
                    self.conv2.weight.requires_grad = False

                    out = F.relu(self.bn1(x))
                    out = self.shortcut(out) if hasattr(self, 'shortcut') else x
                return out
            else:

                if torch.equal(m_sample_list[8 - all_block_count], torch.ones(1)):
                    all_block_count = all_block_count - 1
                    print("******************")
                    self.conv1.weight.requires_grad = True
                    self.conv2.weight.requires_grad = True
                    out = F.relu(self.bn1(x))
                    shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
                    out = self.conv1(out)
                    out = self.conv2(F.relu(self.bn2(out)))
                    out += shortcut
                else:
                    print("!!!!!!!!!!!!!!!!!!")
                    all_block_count = all_block_count - 1
                    self.conv1.weight.requires_grad = False
                    self.conv2.weight.requires_grad = False

                    out = F.relu(self.bn1(x))
                    out = self.shortcut(out) if hasattr(self, 'shortcut') else x

            return out
        else:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet_sto(nn.Module):
    def __init__(self, block,prob_0_L, num_blocks, num_classes=200):
        super(PreActResNet_sto, self).__init__()
        self.in_planes = 64
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta / (sum(num_blocks) - 1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*4, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.prob_now,self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            self.prob_now = self.prob_now - self.prob_step
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/apdcephfs/share_1290939/jiaxiaojun/tiny-imagenet-200')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='/apdcephfs/share_1290939/jiaxiaojun/FGSM_SD_AT/Tiny_Imagenet/FGSM_DSD/epochs_110/lr_max_0.1/model_PreActResNest18/lr_schedule_multistep/alpha_10.0/prob_0_1.0/prob_1_0.5/factor_0.6/best_model.pth')
    parser.add_argument('--model', default='PreActResNest18', type=str, help='model name')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--out_dir', type=str, default='./data')

    arguments = parser.parse_args()
    return arguments


args = get_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
logfile1 = os.path.join(args.out_dir, 'log_file1.txt')

if os.path.exists(logfile1):
    os.remove(logfile1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

target_model=PreActResNet_sto(PreActBlock, prob_0_L=[1, 1], num_blocks = [2,2,2,2])
target_model = target_model.to(device)
checkpoint = torch.load(args.model_path)



target_model.load_state_dict(checkpoint)


target_model.eval()
train_loader, test_loader = New_ImageNet_get_loaders_64(args.data_dir, args.batch_size)
epsilon = args.epsilon
epsilon = float(epsilon) / 255.
print(epsilon)


AT_fgsm_loss,AT_fgsm_acc=evaluate_fgsm(test_loader, target_model, 1)
AT_pgd_loss_10, AT_pgd_acc_10 = evaluate_pgd(test_loader, target_model, 10, 1, epsilon / std)
AT_pgd_loss_20, AT_pgd_acc_20 = evaluate_pgd(test_loader, target_model, 20, 1, epsilon / std)
AT_pgd_loss_50, AT_pgd_acc_50 = evaluate_pgd(test_loader, target_model, 50, 1, epsilon / std)

AT_CW_loss_20, AT_pgd_cw_acc_20 = evaluate_pgd_cw(test_loader, target_model, 20, 1)


AT_models_test_loss, AT_models_test_acc = evaluate_standard(test_loader, target_model)

print('AT_models_test_acc:', AT_models_test_acc)
print('AT_fgsm_acc:', AT_fgsm_acc)
print('AT_pgd_acc_10:', AT_pgd_acc_10)
print('AT_pgd_acc_20:', AT_pgd_acc_20)
print('AT_pgd_acc_50:', AT_pgd_acc_50)
print('AT_pgd_cw_acc_20:', AT_pgd_cw_acc_20)

adversary1 = AutoAttack(target_model, norm=args.norm, eps=epsilon, version='standard', log_path=logfile1)

#adversary2 = AutoAttack(target_model, norm=args.norm, eps=epsilon, version='standard', log_path=logfile2)
l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)

adv_complete = adversary1.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                                                  bs=args.batch_size)
# adv_complete1 = adversary2.run_standard_evaluation_individual(x_test[:args.n_ex], y_test[:args.n_ex],
#                                                               bs=args.batch_size)