import torch
import torch.nn.functional as F
from utils.stylegan import create_image
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

def scores_by_transform(imgs,
                        targets,
                        target_model,
                        transforms,
                        iterations=100):

    score = torch.zeros_like(
        targets, dtype=torch.float32).to(imgs.device)

    with torch.no_grad():
        for i in range(iterations):
            imgs_transformed = transforms(imgs)
            prediction_vector = target_model(imgs_transformed).softmax(dim=1)
            score += torch.gather(prediction_vector, 1,
                                  targets.unsqueeze(1)).squeeze()
        score = score / iterations
    return score


def perform_final_selection(w, generator, config, targets, target_model, samples_per_target,
                            approach, iterations, batch_size, device, vae=None, unet=None, rtpt=None):
    target_values = set(targets.cpu().tolist())
    final_targets = []
    final_w = []
    final_imgs = [] #***
    target_model.eval()

    if approach.strip() == 'transforms':
        transforms_ = T.Compose([
            T.RandomResizedCrop(size=(224, 224),
                                scale=(0.5, 0.9),
                                ratio=(0.8, 1.2)),
            T.RandomHorizontalFlip(0.5)
        ])

    for step, target in enumerate(target_values):
        mask = torch.where(targets == target, True, False)
        w_masked = w[mask]
        candidates = create_image(w_masked,
                                  generator,
                                  crop_size=config.attack_center_crop,
                                  resize=config.attack_resize,
                                  device=device).cpu()
        targets_masked = targets[mask].cpu()
        scores = []
        scores_ = []
        imgs_tmp = []
        dataset = TensorDataset(candidates, targets_masked)
        gaijin = 0
        for imgs, t in DataLoader(dataset, batch_size=batch_size):
            imgs, t = imgs.to(device), t.to(device)
            scores.append(scores_by_transform(imgs,
                                                  t,
                                                  target_model,
                                                  transforms_,
                                                  iterations))
            if gaijin == 11:
                from attacks.optimize import RandomMaskingGenerator
                mask_ratio = 0
                masked_position_generator = RandomMaskingGenerator(14, mask_ratio)
                bool_masked_pos = masked_position_generator()
                bool_masked_pos = torch.from_numpy(bool_masked_pos).to(device)
                bool_masked_pos = bool_masked_pos.repeat(imgs.shape[0],1)
                from unilm.beit.dall_e.utils import map_pixels
                visual_token_transform = transforms.Compose([map_pixels,])
                common_transform = transforms.Compose([
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(p=0.5),
                ])
                for_visual_tokens = common_transform(imgs)
                samples = visual_token_transform(for_visual_tokens)
                with torch.no_grad():
                    outputs_beit_mask,  outputs_beit_all = vae(samples, bool_masked_pos=bool_masked_pos.bool(), return_all_tokens=True)
                    recons_img = unet(outputs_beit_all)
                scores_.append(scores_by_transform(recons_img,
                                                      t,
                                                      target_model,
                                                      transforms_,
                                                      iterations))
            print(imgs.shape, '*********************')
            if gaijin == 11:
                tmp = torch.cat((imgs.cpu(), recons_img.cpu()), 0)
                imgs_tmp.append(tmp)
            
                
        if gaijin == 11:
            scores = scores + scores_
            targets_masked = targets_masked.repeat(2)
            w_masked = w_masked.repeat(2,1,1)
            imgs_tmp = torch.cat(imgs_tmp, dim=0).cpu()
            
        scores = torch.cat(scores, dim=0).cpu()
        
       
        indices = torch.sort(scores, descending=True).indices
        selected_indices = indices[:samples_per_target]
        final_targets.append(targets_masked[selected_indices].cpu())
        final_w.append(w_masked[selected_indices].cpu())
        if gaijin == 11:
            final_imgs.append(imgs_tmp[selected_indices].cpu())
        else:
            final_imgs = None
           
        if rtpt:
            rtpt.step(
                subtitle=f'Sample Selection step {step} of {len(target_values)}')
    final_targets = torch.cat(final_targets, dim=0)
    final_w = torch.cat(final_w, dim=0)
    if gaijin == 11:
        final_imgs = torch.cat(final_imgs, dim=0)
        return final_w, final_targets, final_imgs
    else:
        return final_w, final_targets
    
