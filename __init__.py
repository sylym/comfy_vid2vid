import os
import torch
import re
import numpy as np
from PIL import Image
from comfy import model_management
import comfy.samplers
from .sd import load_checkpoint_guess_config
import comfy.utils
import folder_paths


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    noise_mask = None
    device = model_management.get_torch_device()

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=torch.manual_seed(seed), device="cpu")

    if "noise_mask_sequence" in latent:
        noise_mask_list = []
        for noise_mask in latent["noise_mask_sequence"]:
            noise_mask = torch.nn.functional.interpolate(noise_mask[None,None,], size=(noise.shape[2], noise.shape[3]), mode="bilinear")
            noise_mask = noise_mask.round()
            noise_mask = torch.cat([noise_mask] * noise.shape[1], dim=1)
            noise_mask_list.append(noise_mask.squeeze())
        noise_mask = torch.stack(noise_mask_list)
        noise_mask = noise_mask.to(device)


    real_model = None
    model_management.load_model_gpu(model)
    real_model = model.model

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = []
    negative_copy = []

    control_nets = []
    for p in positive:
        t = p[0]
        if t.shape[0] < noise.shape[0]:
            t = torch.cat([t] * noise.shape[0])
        t = t.to(device)
        if 'control' in p[1]:
            control_nets += [p[1]['control']]
        positive_copy += [[t] + p[1:]]
    for n in negative:
        t = n[0]
        if t.shape[0] < noise.shape[0]:
            t = torch.cat([t] * noise.shape[0])
        t = t.to(device)
        if 'control' in n[1]:
            control_nets += [n[1]['control']]
        negative_copy += [[t] + n[1:]]

    control_net_models = []
    for x in control_nets:
        control_net_models += x.get_control_models()
    model_management.load_controlnet_gpu(control_net_models)

    if sampler_name in comfy.samplers.KSampler.SAMPLERS:
        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise)
    else:
        #other samplers
        pass

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask)
    samples = samples.cpu()
    for c in control_nets:
        c.cleanup()

    out = latent.copy()
    out["samples"] = samples
    return (out, )


class KSamplerSequence:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)


class LoadImageSequence:
    input_dir = os.path.realpath("input")
    @classmethod
    def INPUT_TYPES(s):
        if not os.path.exists(s.input_dir):
            os.makedirs(s.input_dir)
        image_folder = [name for name in os.listdir(s.input_dir) if os.path.isdir(os.path.join(s.input_dir,name)) and len(os.listdir(os.path.join(s.input_dir,name))) != 0]
        return {"required":
                    {"image_sequence_folder": (sorted(image_folder), )}
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK_SEQUENCE")
    FUNCTION = "load_image_sequence"

    def load_image_sequence(self, image_sequence_folder):
        image_path = os.path.join(self.input_dir, image_sequence_folder)
        file_list = sorted(os.listdir(image_path), key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))
        sample_frames = []
        sample_frames_mask = []
        for file in file_list:
            i = Image.open(os.path.join(image_path, file))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            image = image.squeeze()
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            sample_frames.append(image)
            sample_frames_mask.append(mask)
        return (torch.stack(sample_frames), sample_frames_mask)


class VAEEncodeForInpaintSequence:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pixels": ("IMAGE", ), "vae": ("VAE", ), "mask_sequence": ("MASK_SEQUENCE", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent/inpaint"

    def encode(self, vae, pixels, mask_sequence):
        pixels_list = [x for x in torch.split(pixels, 1)]
        samples_list = []
        noise_mask_list = []
        if len(pixels_list) != len(mask_sequence):
            raise Exception("Number of pixels and masks must match")
        for num in range(len(pixels_list)):
            pixels = pixels_list[num]
            mask = mask_sequence[num]
            x = (pixels.shape[1] // 64) * 64
            y = (pixels.shape[2] // 64) * 64
            mask = torch.nn.functional.interpolate(mask[None,None,], size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")[0][0]

            pixels = pixels.clone()
            if pixels.shape[1] != x or pixels.shape[2] != y:
                pixels = pixels[:,:x,:y,:]
                mask = mask[:x,:y]

            #grow mask by a few pixels to keep things seamless in latent space
            kernel_tensor = torch.ones((1, 1, 6, 6))
            mask_erosion = torch.clamp(torch.nn.functional.conv2d((mask.round())[None], kernel_tensor, padding=3), 0, 1)
            noise_mask_list.append((mask_erosion[0][:x,:y].round()))
            m = (1.0 - mask.round())
            for i in range(3):
                pixels[:,:,:,i] -= 0.5
                pixels[:,:,:,i] *= m
                pixels[:,:,:,i] += 0.5
            samples_list.append(vae.encode(pixels).squeeze())

        return ({"samples":torch.stack(samples_list),"noise_mask_sequence":noise_mask_list},)


class LoadImageMaskSequence:
    input_dir = os.path.realpath("input")
    @classmethod
    def INPUT_TYPES(s):
        if not os.path.exists(s.input_dir):
            os.makedirs(s.input_dir)
        image_folder = [name for name in os.listdir(s.input_dir) if os.path.isdir(os.path.join(s.input_dir, name)) and len(os.listdir(os.path.join(s.input_dir, name))) != 0]
        return {"required":
                    {"image_sequence_folder": (sorted(image_folder), ),
                    "channel": (["alpha", "red", "green", "blue"], ),}
                }

    CATEGORY = "image"

    RETURN_TYPES = ("MASK_SEQUENCE",)
    FUNCTION = "load_image_sequence"

    def load_image_sequence(self, image_sequence_folder, channel):
        image_path = os.path.join(self.input_dir, image_sequence_folder)
        file_list = sorted(os.listdir(image_path), key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))
        sample_frames_mask = []
        for file in file_list:
            i = Image.open(os.path.join(image_path, file))
            mask = None
            c = channel[0].upper()
            if c in i.getbands():
                mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask)
                if c == 'A':
                    mask = 1. - mask
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            sample_frames_mask.append(mask)
        return (sample_frames_mask,)


class CheckpointLoaderSimpleSequence:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out


NODE_CLASS_MAPPINGS = {
    "LoadImageSequence": LoadImageSequence,
    "VAEEncodeForInpaintSequence": VAEEncodeForInpaintSequence,
    "KSamplerSequence": KSamplerSequence,
    "LoadImageMaskSequence": LoadImageMaskSequence,
    "CheckpointLoaderSimpleSequence": CheckpointLoaderSimpleSequence,
}
