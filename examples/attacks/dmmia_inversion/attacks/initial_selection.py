import numpy as np
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
from utils.stylegan import adjust_gen_images


def find_initial_w(generator,
                   target_model,
                   targets,
                   search_space_size,
                   clip=True,
                   center_crop=768,
                   resize=224,
                   horizontal_flip=True,
                   filepath=None,
                   truncation_psi=0.7,
                   truncation_cutoff=18,
                   batch_size=25,
                   seed=0,
                   device='cuda:4'):
    """Find good initial starting points in the style space.

    Args:
        generator (torch.nn.Module): StyleGAN2 model
        target_model (torch.nn.Module): [description]
        target_cls (int): index of target class.
        search_space_size (int): number of potential style vectors.
        clip (boolean, optional): clip images to [-1, 1]. Defaults to True.
        center_crop (int, optional): size of the center crop. Defaults 768.
        resize (int, optional): size for the resizing operation. Defaults to 224.
        horizontal_flip (boolean, optional): apply horizontal flipping to images. Defaults to true.
        filepath (str): filepath to save candidates.
        truncation_psi (float, optional): truncation factor. Defaults to 0.7.
        truncation_cutoff (int, optional): truncation cutoff. Defaults to 18.
        batch_size (int, optional): batch size. Defaults to 25.

    Returns:
        torch.tensor: style vectors with highest confidences on the target model and target class.
    """
    #device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    z = torch.from_numpy(
        np.random.RandomState(seed).randn(search_space_size,
                                          generator.z_dim)).to(device)
    c = None
    target_model.eval()
    five_crop = None
    with torch.no_grad():
        confidences = []
        final_candidates = []
        final_confidences = []
        candidates = generator.mapping(z,
                                       c,
                                       truncation_psi=truncation_psi,
                                       truncation_cutoff=truncation_cutoff)
        candidate_dataset = torch.utils.data.TensorDataset(candidates)
        for w in tqdm(torch.utils.data.DataLoader(candidate_dataset,
                                                  batch_size=batch_size),
                      desc='Find initial style vector w'):
            imgs = generator.synthesis(w[0],
                                       noise_mode='const',
                                       force_fp32=True)
            # Adjust images and perform augmentation
            if clip:
                lower_bound = torch.tensor(-1.0).float().to(imgs.device)
                upper_bound = torch.tensor(1.0).float().to(imgs.device)
                imgs = torch.where(imgs > upper_bound, upper_bound, imgs)
                imgs = torch.where(imgs < lower_bound, lower_bound, imgs)
            if center_crop is not None:
                imgs = F.center_crop(imgs, (center_crop, center_crop))
            if resize is not None:
                imgs = [F.resize(imgs, resize)]
            if horizontal_flip:
                imgs.append(F.hflip(imgs[0]))
            if five_crop:
                cropped_images = []
                for im in imgs:
                    cropped_images += list(five_crop(im))
                imgs = cropped_images
            target_conf = None
            for im in imgs:
                if target_conf is not None:
                    target_conf += target_model(im).softmax(dim=1) / len(imgs)
                else:
                    target_conf = target_model(im).softmax(dim=1) / len(imgs)
            confidences.append(target_conf)

        confidences = torch.cat(confidences, dim=0)
        for target in targets:
            sorted_conf, sorted_idx = confidences[:,
                                                  target].sort(descending=True)
            final_candidates.append(candidates[sorted_idx[0]].unsqueeze(0))
            final_confidences.append(sorted_conf[0].cpu().item())
            # Avoid identical candidates for the same target
            confidences[sorted_idx[0], target] = -1.0

    final_candidates = torch.cat(final_candidates, dim=0).to(device)
    final_confidences = [np.round(c, 2) for c in final_confidences]
    print(
        f'Found {final_candidates.shape[0]} initial style vectors.'
    )

    if filepath:
        torch.save(final_candidates, filepath)
        print(f'Candidates have been saved to {filepath}')
    return final_candidates
