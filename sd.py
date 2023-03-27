import torch
from comfy import model_management
from comfy.sd import load_torch_file, load_model_weights, ModelPatcher, VAE, CLIP
from comfy.ldm.util import instantiate_from_config
from .convert_from_ckpt import convert_unet_checkpoint
from omegaconf import OmegaConf


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
    sd = load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    vae = None

    fp16 = model_management.should_use_fp16()

    class WeightsLoader(torch.nn.Module):
        pass

    w = WeightsLoader()
    load_state_dict_to = []
    if output_vae:
        vae = VAE()
        w.first_stage_model = vae.first_stage_model
        load_state_dict_to = [w]

    if output_clip:
        clip_config = {}
        if "cond_stage_model.model.transformer.resblocks.22.attn.out_proj.weight" in sd_keys:
            clip_config['target'] = 'ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder'
        else:
            clip_config['target'] = 'ldm.modules.encoders.modules.FrozenCLIPEmbedder'
        clip = CLIP(config=clip_config, embedding_directory=embedding_directory)
        w.cond_stage_model = clip.cond_stage_model
        load_state_dict_to = [w]

    sd_config = {
        "linear_start": 0.00085,
        "linear_end": 0.012,
        "num_timesteps_cond": 1,
        "log_every_t": 200,
        "timesteps": 1000,
        "first_stage_key": "jpg",
        "cond_stage_key": "txt",
        "image_size": 64,
        "channels": 4,
        "cond_stage_trainable": False,
        "monitor": "val/loss_simple_ema",
        "scale_factor": 0.18215,
        "use_ema": False,
    }

    unet_config = {
        "use_checkpoint": True,
        "image_size": 32,
        "out_channels": 4,
        "attention_resolutions": [
            4,
            2,
            1
        ],
        "num_res_blocks": 2,
        "channel_mult": [
            1,
            2,
            4,
            4
        ],
        "use_spatial_transformer": True,
        "transformer_depth": 1,
        "legacy": False
    }

    if len(sd['model.diffusion_model.input_blocks.1.1.proj_in.weight'].shape) == 2:
        unet_config['use_linear_in_transformer'] = True

    unet_config["use_fp16"] = fp16
    unet_config["model_channels"] = sd['model.diffusion_model.input_blocks.0.0.weight'].shape[0]
    unet_config["in_channels"] = sd['model.diffusion_model.input_blocks.0.0.weight'].shape[1]
    unet_config["context_dim"] = sd['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight'].shape[1]

    sd_config["unet_config"] = {"target": "ldm.modules.diffusionmodules.openaimodel.UNetModel", "params": unet_config}
    model_config = {"target": "ldm.models.diffusion.ddpm.LatentDiffusion", "params": sd_config}

    if unet_config["in_channels"] > 4: #inpainting model
        sd_config["conditioning_key"] = "hybrid"
        sd_config["finetune_keys"] = None
        model_config["target"] = "ldm.models.diffusion.ddpm.LatentInpaintDiffusion"
    else:
        sd_config["conditioning_key"] = "crossattn"

    if unet_config["context_dim"] == 1024:
        unet_config["num_head_channels"] = 64 #SD2.x
    else:
        unet_config["num_heads"] = 8 #SD1.x

    if unet_config["context_dim"] == 1024 and unet_config["in_channels"] == 4: #only SD2.x non inpainting models are v prediction
        k = "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias"
        out = sd[k]
        if torch.std(out, unbiased=False) > 0.09: # not sure how well this will actually work. I guess we will find out.
            sd_config["parameterization"] = 'v'

    model = instantiate_from_config(model_config)
    model = load_model_weights(model, sd, verbose=False, load_state_dict_to=load_state_dict_to)
    with torch.inference_mode(mode=False):
        model.model.diffusion_model = convert_unet_checkpoint(sd, OmegaConf.create({"model": model_config}))

    if model_management.xformers_enabled():
        model.model.diffusion_model.enable_xformers_memory_efficient_attention()

    #if fp16:
    #   model = model.half()

    return (ModelPatcher(model), clip, vae)
