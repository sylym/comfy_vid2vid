import os
import shutil
import numpy as np
from typing import Union
from PIL import Image
import re
import torch

from tqdm import tqdm
from einops import rearrange
from safetensors.torch import load_file
import cv2
import subprocess
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from controlnet_aux import HEDdetector, MLSDdetector, OpenposeDetector
from .controlnet_utils import ade_palette


def save_videos_grid(videos: torch.Tensor, path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    videos = rearrange(videos, "b c t h w -> t b c h w")
    num = 0
    for x in videos:
        image = x.squeeze().numpy()
        image = np.transpose(image[[2, 1, 0], :, :], (1, 2, 0))
        image = (image * 65535.0).round().astype(np.uint16)
        cv2.imwrite(os.path.join(path, f"{os.path.basename(path)}_{num}.png"), image)
        num += 1


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipe):
    uncond_input = pipe.tokenizer(
        [""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
    text_input = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipe, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipe)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipe.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipe, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipe, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def use_lora(pretrained_LoRA_path, pipe, alpha):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    state_dict = load_file(pretrained_LoRA_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:

        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipe.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipe.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)
    return pipe


def down_up_sample(input_frames_folder, down_sample, no_down=False):
    if down_sample == 2 or down_sample == 4:
        temp_frames_folder = input_frames_folder + "_temp"
        if os.path.exists(temp_frames_folder):
            shutil.rmtree(temp_frames_folder)
        if no_down:
            print("up sample with x" + str(down_sample))
            subprocess.run(["python", "./Real-ESRGAN/inference_realesrgan.py", "-i", input_frames_folder, "-o", temp_frames_folder, "-s", str(down_sample), "--fp32"], check=True)
            shutil.rmtree(input_frames_folder)
            os.rename(os.path.abspath(temp_frames_folder), os.path.abspath(input_frames_folder))
        else:
            print("down and up sample with x" + str(down_sample))
            subprocess.run(["python", "./Real-ESRGAN/inference_realesrgan.py", "-i", input_frames_folder, "-o", temp_frames_folder, "-s", str(down_sample), "--fp32"], check=True)
            shutil.rmtree(input_frames_folder)
            for file in os.listdir(temp_frames_folder):
                img = cv2.imread(os.path.join(temp_frames_folder, file), cv2.IMREAD_UNCHANGED)
                new_size = (img.shape[1] // 2, img.shape[0] // 2)
                resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(os.path.join(temp_frames_folder, file), resized_img)
            os.rename(os.path.abspath(temp_frames_folder), os.path.abspath(input_frames_folder))
    else:
        print("do noting with down_up_sample")


def merge_frames(video_list, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    content_list = {}
    all_frames_path_list = []
    for folder in video_list:
        content_list[folder] = os.listdir(folder)
    for key, value in content_list.items():
        for file in value:
            source_path = os.path.join(key, file)
            destination_path = os.path.join(output_path, file)
            shutil.copy(source_path, destination_path)
            all_frames_path_list.append(destination_path)

    all_frames_path_list = sorted(all_frames_path_list, key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))

    for i, file_path in enumerate(all_frames_path_list):
        dir_name = os.path.dirname(file_path)
        ext_name = os.path.splitext(file_path)[1]
        new_file_name = str(i + 1) + ext_name
        new_file_path = os.path.join(dir_name, new_file_name)
        os.rename(file_path, new_file_path)



def controlnet_image_preprocessing(image_list, video_prepare_type):
    image_list_out = []
    if video_prepare_type == "canny":
        for image in image_list:
            image = np.array(image)
            low_threshold = 100
            high_threshold = 200
            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image)
            image_list_out.append(image)

    elif video_prepare_type == "depth":
        depth_estimator = pipeline('depth-estimation')
        for image in image_list:
            image = depth_estimator(image)['depth']
            image = np.array(image)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image)
            image_list_out.append(image)

    elif video_prepare_type == "mlsd":
        mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        for image in image_list:
            image = mlsd(image)
            image_list_out.append(image)

    elif video_prepare_type == "hed":
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        for image in image_list:
            image = hed(image)
            image_list_out.append(image)

    elif video_prepare_type == "normal":
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        for image in image_list:
            image = image.convert("RGB")
            image = depth_estimator(image)['predicted_depth'][0]
            image = image.numpy()
            image_depth = image.copy()
            image_depth -= np.min(image_depth)
            image_depth /= np.max(image_depth)
            bg_threhold = 0.4
            x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            x[image_depth < bg_threhold] = 0
            y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            y[image_depth < bg_threhold] = 0
            z = np.ones_like(x) * np.pi * 2.0
            image = np.stack([x, y, z], axis=2)
            image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
            image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image)
            image_list_out.append(image)

    elif video_prepare_type == "openpose":
        openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        for image in image_list:
            image = openpose(image)
            image_list_out.append(image)

    elif video_prepare_type == "scribble":
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        for image in image_list:
            image = hed(image, scribble=True)
            image_list_out.append(image)

    elif video_prepare_type == "seg":
        image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        for image in image_list:
            image = image.convert('RGB')
            pixel_values = image_processor(image, return_tensors="pt").pixel_values
            with torch.no_grad():
                outputs = image_segmentor(pixel_values)
            seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
            palette = np.array(ade_palette())
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color
            color_seg = color_seg.astype(np.uint8)
            image = Image.fromarray(color_seg)
            image_list_out.append(image)

    else:
        image_list_out = image_list

    os.makedirs("./temp/input_image", exist_ok=True)
    for image_num in range(len(image_list_out)):
        image_list_out[image_num].save(f"./temp/input_image/{video_prepare_type}_{str(image_num)}.png")

    return image_list_out
