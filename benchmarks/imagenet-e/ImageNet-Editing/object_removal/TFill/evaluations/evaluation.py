import os
import glob
import shutil
import lpips
import numpy as np
import argparse
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from dataloader.image_folder import make_dataset
from util import util
import torch

parser = argparse.ArgumentParser(description='Image quality evaluations on the dataset')
parser.add_argument('--gt_path', type=str, default='../results/', help='path to original gt data')
parser.add_argument('--g_path', type=str, default='../results.', help='path to the generated data')
parser.add_argument('--save_path', type=str, default=None, help='path to save the best results')
parser.add_argument('--center', action='store_true', help='only calculate the center masked regions for the image quality')
parser.add_argument('--num_test', type=int, default=0, help='how many examples to load for testing')

args = parser.parse_args()
lpips_alex = lpips.LPIPS(net='alex')


def calculate_score(img_gt, img_test):
    """
    function to calculate the image quality score
    :param img_gt: original image
    :param img_test: generated image
    :return: mae, ssim, psnr
    """

    l1loss = np.mean(np.abs(img_gt-img_test))

    psnr_score = psnr(img_gt, img_test, data_range=1)

    ssim_score = ssim(img_gt, img_test, multichannel=True, data_range=1, win_size=11)

    lpips_dis = lpips_alex(torch.from_numpy(img_gt).permute(2, 0, 1), torch.from_numpy(img_test).permute(2, 0, 1), normalize=True)

    return l1loss, ssim_score, psnr_score, lpips_dis.data.numpy().item()


if __name__ == '__main__':
    gt_paths, gt_size = make_dataset(args.gt_path)
    g_paths, g_size = make_dataset(args.g_path)

    l1losses = []
    ssims = []
    psnrs = []
    lpipses = []

    size = args.num_test if args.num_test > 0 else gt_size

    for i in range(size):
        gt_img = Image.open(gt_paths[i + 0*2000]).resize([256, 256]).convert('RGB')
        gt_numpy = np.array(gt_img).astype(np.float32) / 255.0
        if args.center:
            gt_numpy = gt_numpy[64:192, 64:192, :]

        l1loss_sample = 1000
        ssim_sample = 0
        psnr_sample = 0
        lpips_sample = 1000

        name = gt_paths[i + 0*2000].split('/')[-1].split(".")[0] + "*"
        g_paths = sorted(glob.glob(os.path.join(args.g_path, name)))
        num_files = len(g_paths)

        for j in range(num_files):
            index = j
            try:
                g_img = Image.open(g_paths[j]).resize([256, 256]).convert('RGB')
                g_numpy = np.array(g_img).astype(np.float32) / 255.0
                if args.center:
                    g_numpy = g_numpy[64:192, 64:192, :]
                l1loss, ssim_score, psnr_score, lpips_score = calculate_score(gt_numpy, g_numpy)
                if l1loss - ssim_score - psnr_score + lpips_score < l1loss_sample - ssim_sample - psnr_sample + lpips_sample:
                    l1loss_sample, ssim_sample, psnr_sample, lpips_sample = l1loss, ssim_score, psnr_score, lpips_score
                    best_index = index
            except:
                print(g_paths[index])

        if l1loss_sample != 1000 and ssim_sample !=0 and psnr_sample != 0:
            print(g_paths[best_index])
            print(l1loss_sample, ssim_sample, psnr_sample, lpips_sample)
            l1losses.append(l1loss_sample)
            ssims.append(ssim_sample)
            psnrs.append(psnr_sample)
            lpipses.append(lpips_sample)

            if args.save_path is not None:
                util.mkdir(args.save_path)
                shutil.copy(g_paths[best_index], args.save_path)

    print('{:>10},{:>10},{:>10},{:>10}'.format('l1loss', 'SSIM', 'PSNR', 'LPIPS'))
    print('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.mean(l1losses), np.mean(ssims), np.mean(psnrs), np.mean(lpipses)))
    print('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.var(l1losses), np.var(ssims), np.var(psnrs), np.var(lpipses)))