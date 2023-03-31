import os
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils.metrics_accumulator import MetricsAccumulator
from utils.video import save_video
from utils.fft_pytorch import HighFrequencyLoss

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss
import lpips
import numpy as np

from CLIP import clip
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    create_classifier,
    classifier_defaults,
)
from utils.visualization import show_tensor_image, show_editied_masked_image
from utils.change_place import change_place, find_bbox

import pdb
import cv2

def create_classifier_ours():

    model = torchvision.models.resnet50()
    ckpt = torch.load('checkpoints/DRA_resnet50.pth')['model_state_dict']
    model.load_state_dict({k.replace('module.','').replace('last_linear','fc'):v for k,v in ckpt.items()})
    model = torch.nn.Sequential(*[torch.nn.Upsample(size=(256,256)), model])
    return model

class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(os.path.join(self.args.output_path, RANKED_RESULTS_DIR))
        os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.export_assets:
            self.assets_path = Path(os.path.join(self.args.output_path, ASSETS_DIR_NAME))
            os.makedirs(self.assets_path, exist_ok=True)
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        self.classifier_config = classifier_defaults()
        self.classifier_config.update(
            {
                "image_size": self.args.model_output_size,
            }
        )

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(
            torch.load(
                "checkpoints/256x256_diffusion_uncond.pt"
                if self.args.model_output_size == 256
                else "checkpoints/512x512_diffusion.pt",
                map_location="cpu",
            )
        )
        # self.model.requires_grad_(False).eval().to(self.device)
        self.model.eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

        self.classifier = create_classifier(**self.classifier_config)
        self.classifier.load_state_dict(
            torch.load("checkpoints/256x256_classifier.pt", map_location="cpu")
        )
        # self.classifier.requires_grad_(False).eval().to(self.device)


        # self.classifier = create_classifier_ours()

        self.classifier.eval().to(self.device)
        if self.classifier_config["classifier_use_fp16"]:
            self.classifier.convert_to_fp16()

        self.clip_model = (
            clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False)
        )
        self.clip_size = self.clip_model.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.to_tensor = transforms.ToTensor()
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()

        self.hf_loss = HighFrequencyLoss()


    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep


    def clip_loss(self, x_in, text_embed):
        clip_loss = torch.tensor(0)

        if self.mask is not None:
            masked_input = x_in * self.mask
        else:
            masked_input = x_in
        augmented_input = self.image_augmentations(masked_input).add(1).div(2) # shape: [N,C,H,W], range: [0,1]
        clip_in = self.clip_normalize(augmented_input)
        # pdb.set_trace()
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = d_clip_loss(image_embeds, text_embed)

        # We want to sum over the averages
        for i in range(self.args.batch_size):
            # We want to average at the "augmentations level"
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss

    def unaugmented_clip_distance(self, x, text_embed):
        x = F.resize(x, [self.clip_size, self.clip_size])
        image_embeds = self.clip_model.encode_image(x).float()
        dists = d_clip_loss(image_embeds, text_embed)

        return dists.item()

    def model_fn(self, x,t,y=None):
        return self.model(x, t, y if self.args.class_cond else None)

    def edit_image_by_prompt(self):
        if self.args.image_guide:
            img_guidance = Image.open(self.args.prompt).convert('RGB')
            img_guidance = img_guidance.resize((224,224), Image.LANCZOS)  # type: ignore
            img_guidance = self.clip_normalize(self.to_tensor(img_guidance).unsqueeze(0)).to(self.device)
            text_embed = self.clip_model.encode_image(img_guidance).float()

        else:
            text_embed = self.clip_model.encode_text(
                clip.tokenize(self.args.prompt).to(self.device)
            ).float()

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        self.init_image_pil_2 = Image.open(self.args.init_image_2).convert("RGB")
        if self.args.rotate_obj:
            # angle = random.randint(-45,45)
            angle = self.args.angle
            self.init_image_pil_2 = self.init_image_pil_2.rotate(angle)
        self.init_image_pil_2 = self.init_image_pil_2.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image_2 = (
            TF.to_tensor(self.init_image_pil_2).to(self.device).unsqueeze(0).mul(2).sub(1)
        )

        '''
        # Init with the inpainting image
        self.init_image_pil_ = Image.open('output/ImageNet-S_val/bad_case_RN50/ILSVRC2012_val_00013212/ranked/08480_output_i_0_b_0.png').convert("RGB")
        self.init_image_pil_ = self.init_image_pil_.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image_ = (
            TF.to_tensor(self.init_image_pil_).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        '''

        if self.args.export_assets:
            img_path = self.assets_path / Path(self.args.output_file)
            self.init_image_pil.save(img_path, quality=100)

        self.mask = torch.ones_like(self.init_image, device=self.device)
        self.mask_pil = None
        if self.args.mask is not None:
            self.mask_pil = Image.open(self.args.mask).convert("RGB")
            if self.args.rotate_obj:
                self.mask_pil = self.mask_pil.rotate(angle)
            if self.mask_pil.size != self.image_size:
                self.mask_pil = self.mask_pil.resize(self.image_size, Image.NEAREST)  # type: ignore
            if self.args.random_position:
                bbox = find_bbox(np.array(self.mask_pil))
                print(bbox)

            image_mask_pil_binarized = ((np.array(self.mask_pil) > 0.5) * 255).astype(np.uint8)
            # image_mask_pil_binarized = cv2.dilate(image_mask_pil_binarized, np.ones((50,50), np.uint8), iterations=1)
            if self.args.invert_mask:
                image_mask_pil_binarized = 255 - image_mask_pil_binarized
                self.mask_pil = TF.to_pil_image(image_mask_pil_binarized)
            self.mask = TF.to_tensor(Image.fromarray(image_mask_pil_binarized))
            self.mask = self.mask[0, ...].unsqueeze(0).unsqueeze(0).to(self.device)
            # self.mask[:] = 1

            if self.args.random_position:
                # print(self.init_image_2.shape, self.init_image_2.max(), self.init_image_2.min())
                # print(self.mask.shape, self.mask.max(), self.mask.min())
                # cv2.imwrite('tmp/init_before.jpg', np.transpose(((self.init_image_2+1)/2*255).cpu().numpy()[0], (1,2,0))[:,:,::-1])
                # cv2.imwrite('tmp/mask_before.jpg', (self.mask*255).cpu().numpy()[0][0])
                self.init_image_2, self.mask = change_place(self.init_image_2, self.mask, bbox, self.args.invert_mask)
                # cv2.imwrite('tmp/init_after.jpg', np.transpose(((self.init_image_2+1)/2*255).cpu().numpy()[0], (1,2,0))[:,:,::-1])
                # cv2.imwrite('tmp/mask_after.jpg', (self.mask*255).cpu().numpy()[0][0])

            if self.args.export_assets:
                mask_path = self.assets_path / Path(
                    self.args.output_file.replace(".png", "_mask.png")
                )
                self.mask_pil.save(mask_path, quality=100)

        def class_guided(x, y, t):
            assert y is not None
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                # logits = self.classifier(x_in, t)
                logits = self.classifier(x_in)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                loss = selected.sum()

                return -torch.autograd.grad(loss, x_in)[0] * self.args.classifier_scale

        def cond_fn(x, t, y=None):
            if self.args.prompt == "":
                return torch.zeros_like(x)
            # pdb.set_trace()
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                t_unscale = self.unscale_timestep(t)

                '''
                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                '''
                out = self.diffusion.p_mean_variance(
                    self.model, x, t_unscale, clip_denoised=False, model_kwargs={"y": None}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t_unscale[0].item()]
                # x_in = out["pred_xstart"] * fac + x * (1 - fac)
                x_in = out["pred_xstart"] # Revised by XX, 2022.07.14

                loss = torch.tensor(0)
                if self.args.classifier_scale != 0 and y is not None:
                    # gradient_class_guided = class_guided(x, y, t)
                    gradient_class_guided = class_guided(x_in, y, t)

                if self.args.background_complex != 0:
                    if self.args.hard:
                        loss = loss - self.args.background_complex*self.hf_loss((x_in+1.)/2.)
                    else:
                        loss = loss + self.args.background_complex*self.hf_loss((x_in+1.)/2.)

                if self.args.clip_guidance_lambda != 0:
                    clip_loss = self.clip_loss(x_in, text_embed) * self.args.clip_guidance_lambda
                    loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())

                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                if self.args.background_preservation_loss:
                    x_in = out["pred_xstart"] * fac + x * (1 - fac)
                    if self.mask is not None:
                        # masked_background = x_in * (1 - self.mask)
                        masked_background = x_in * self.mask # 2022.07.19
                    else:
                        masked_background = x_in

                    if self.args.lpips_sim_lambda:
                        '''
                        loss = (
                            loss
                            + self.lpips_model(masked_background, self.init_image).sum()
                            * self.args.lpips_sim_lambda
                        )
                        '''
                        # 2022.07.19
                        loss = (
                            loss
                            + self.lpips_model(masked_background, self.init_image*self.mask).sum()
                            * self.args.lpips_sim_lambda
                        )
                    if self.args.l2_sim_lambda:
                        '''
                        loss = (
                            loss
                            + mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda
                        )
                        '''
                        # 2022.07.19
                        loss = (
                            loss
                            + mse_loss(masked_background, self.init_image*self.mask) * self.args.l2_sim_lambda
                        )


                if self.args.classifier_scale != 0 and y is not None:
                    return -torch.autograd.grad(loss, x)[0] + gradient_class_guided
                else:
                    return -torch.autograd.grad(loss, x)[0]

        @torch.no_grad()
        def postprocess_fn(out, t):
            if self.args.coarse_to_fine:
                if t > 50:
                    kernel = 51
                elif t > 35:
                    kernel = 31
                else:
                    kernel = 0
                if kernel > 0:
                    max_pool = torch.nn.MaxPool2d(kernel_size=kernel, stride=1, padding=int((kernel-1)/2))
                    self.mask_d = 1 - self.mask
                    self.mask_d = max_pool(self.mask_d)
                    self.mask_d = 1 - self.mask_d
                else:
                    self.mask_d = self.mask
            else:
                self.mask_d = self.mask

            if self.mask is not None:
                background_stage_t = self.diffusion.q_sample(self.init_image_2, t[0])
                background_stage_t = torch.tile(
                    background_stage_t, dims=(self.args.batch_size, 1, 1, 1)
                )
                out["sample"] = out["sample"] * self.mask_d + background_stage_t * (1 - self.mask_d)

            return out

        save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number}")

            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
            )
            samples = sample_func(
                self.model_fn,
                (
                    self.args.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                # model_kwargs={}
                # if self.args.model_output_size == 256
                # else {
                #     "y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
                # },
                model_kwargs={}
                if self.args.classifier_scale == 0
                else {"y": self.args.y*torch.ones([self.args.batch_size], device=self.device, dtype=torch.long)},
                cond_fn=cond_fn,
                device=self.device,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image,
                # init_image=self.init_image_,
                postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
                randomize_class=True if self.args.classifier_scale == 0 else False,
            )

            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps
                if should_save_image or self.args.save_video:
                    self.metrics_accumulator.print_average_metric()

                    for b in range(self.args.batch_size):
                        pred_image = sample["pred_xstart"][b]
                        visualization_path = Path(
                            os.path.join(self.args.output_path, self.args.output_file)
                        )
                        visualization_path = visualization_path.with_stem(
                            f"{visualization_path.stem}_i_{iteration_number}_b_{b}"
                        )
                        if (
                            self.mask is not None
                            and self.args.enforce_background
                            and j == total_steps
                            and not self.args.local_clip_guided_diffusion
                        ):
                            pred_image = (
                                self.init_image_2[0] * (1 - self.mask[0]) + pred_image * self.mask[0]
                            )
                        '''
                        if j == total_steps:
                            pdb.set_trace()
                        pred_image = (
                                self.init_image_2[0] * (1 - self.mask[0]) + pred_image * self.mask[0]
                            )
                        '''
                        pred_image = pred_image.add(1).div(2).clamp(0, 1)
                        pred_image_pil = TF.to_pil_image(pred_image)
                        masked_pred_image = self.mask * pred_image.unsqueeze(0)
                        final_distance = self.unaugmented_clip_distance(
                            masked_pred_image, text_embed
                        )
                        formatted_distance = f"{final_distance:.4f}"

                        if self.args.export_assets:
                            pred_path = self.assets_path / visualization_path.name
                            pred_image_pil.save(pred_path, quality=100)

                        if j == total_steps:
                            path_friendly_distance = formatted_distance.replace(".", "")
                            ranked_pred_path = self.ranked_results_path / (
                                path_friendly_distance + "_" + visualization_path.name
                            )
                            pred_image_pil.save(ranked_pred_path, quality=100)

                        intermediate_samples[b].append(pred_image_pil)
                        if should_save_image:
                            show_editied_masked_image(
                                title=self.args.prompt,
                                source_image=self.init_image_pil,
                                edited_image=pred_image_pil,
                                mask=self.mask_pil,
                                path=visualization_path,
                                distance=formatted_distance,
                            )

            if self.args.save_video:
                for b in range(self.args.batch_size):
                    video_name = self.args.output_file.replace(
                        ".png", f"_i_{iteration_number}_b_{b}.avi"
                    )
                    video_path = os.path.join(self.args.output_path, video_name)
                    save_video(intermediate_samples[b], video_path)

        visualize_size = (256,256)
        img_ori = cv2.imread(self.args.init_image_2) 
        img_ori = cv2.resize(img_ori, visualize_size)
        mask = cv2.imread(self.args.mask)
        mask = cv2.resize(mask, visualize_size)
        imgs = [img_ori, mask]
        for ii, img_name in enumerate(os.listdir(os.path.join(self.args.output_path, 'ranked'))):
            img_path = os.path.join(self.args.output_path, 'ranked', img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, visualize_size)
            imgs.append(img)
            if ii >= 7:
                break

        img_whole = cv2.hconcat(imgs[2:])
        '''
        img_name = self.args.output_path.split('/')[-2]+'/'
        if self.args.coarse_to_fine:
            if self.args.clip_guidance_lambda == 0:
                prompt = 'coarse_to_fine_no_clip'
            else:
                prompt = 'coarse_to_fine'
        elif self.args.image_guide:
            prompt = 'image_guide'
        elif self.args.clip_guidance_lambda == 0:
            prompt = 'no_clip_guide'
        else:
            prompt = 'text_guide'
        '''

        cv2.imwrite(os.path.join(self.args.final_save_root, 'edited.png'), img_whole, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


    def reconstruct_image(self):
        init = Image.open(self.args.init_image).convert("RGB")
        init = init.resize(
            self.image_size,  # type: ignore
            Image.LANCZOS,
        )
        init = TF.to_tensor(init).to(self.device).unsqueeze(0).mul(2).sub(1)

        samples = self.diffusion.p_sample_loop_progressive(
            self.model,
            (1, 3, self.model_config["image_size"], self.model_config["image_size"],),
            clip_denoised=False,
            model_kwargs={}
            if self.args.model_output_size == 256
            else {"y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)},
            cond_fn=None,
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=init,
            randomize_class=True,
        )
        save_image_interval = self.diffusion.num_timesteps // 5
        max_iterations = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

        for j, sample in enumerate(samples):
            if j % save_image_interval == 0 or j == max_iterations:
                print()
                filename = os.path.join(self.args.output_path, self.args.output_file)
                TF.to_pil_image(sample["pred_xstart"][0].add(1).div(2).clamp(0, 1)).save(filename)
