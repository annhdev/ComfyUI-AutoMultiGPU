import torch
import logging
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management as mm
from .device_manager import get_available_devices, select_optimal_devices, get_gpu_status
from .autogpu_pytorch import resolve_clip_type, FULL_MODEL_TYPES
from .gguf_support import (
    is_gguf_file,
    get_combined_unet_list,
    get_combined_clip_list,
    resolve_unet_path,
    resolve_clip_path,
    load_unet_gguf_on_device,
    load_clip_gguf_on_device,
)

logger = logging.getLogger("AutoMultiGPU")

class AutoUniversalLoaderMultiGPU:
    """
    Universal All-in-One Multi-GPU Loader (Safetensors & GGUF):
    Loads UNet/DiT, CLIP(s), and VAE in a single node with automatic Multi-GPU balancing across ALL model types.
    """
    @classmethod
    def INPUT_TYPES(s):
        from nodes import VAELoader
        unet_files = get_combined_unet_list()
        clip_files = ["none"] + get_combined_clip_list()
        vae_files = ["none"] + VAELoader.vae_list(VAELoader)
        devices = get_available_devices()

        return {
            "required": {
                "unet_name": (unet_files, {"tooltip": "Select UNet / DiT model (.safetensors or .gguf)"}),
                "clip_name1": (clip_files, {"default": clip_files[1] if len(clip_files) > 1 else "none", "tooltip": "First Text Encoder (.safetensors or .gguf)"}),
                "clip_name2": (clip_files, {"default": "none", "tooltip": "Optional Second Text Encoder (e.g. T5 / CLIP-G / Llama)"}),
                "vae_name": (vae_files, {"default": vae_files[1] if len(vae_files) > 1 else "none", "tooltip": "VAE model"}),
                "model_type": (FULL_MODEL_TYPES, {"default": "auto", "tooltip": "Comprehensive Architecture Model Type"}),
                "distribution_mode": ([
                    "Auto: UNet GPU0 / Encoders GPU1",
                    "Auto: Balanced Free VRAM",
                    "Manual Override"
                ], {"default": "Auto: UNet GPU0 / Encoders GPU1"}),
            },
            "optional": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "float16", "bfloat16", "float32"], {"default": "default"}),
                "manual_unet_device": (devices, {"default": "cuda:0" if "cuda:0" in devices else "cpu"}),
                "manual_clip_device": (devices, {"default": "cuda:1" if "cuda:1" in devices else "cpu"}),
                "manual_vae_device": (devices, {"default": "cuda:1" if "cuda:1" in devices else "cpu"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_all"
    CATEGORY = "AutoMultiGPU"
    TITLE = "Auto Universal Loader (MultiGPU - Safetensors & GGUF)"
    DESCRIPTION = "Loads UNet, Dual/Single CLIP, and VAE (Safetensors or GGUF) with zero-configuration multi-GPU distribution across all model types."

    def load_all(self, unet_name, clip_name1, clip_name2, vae_name, model_type="auto",
                 distribution_mode="Auto: UNet GPU0 / Encoders GPU1", weight_dtype="default",
                 manual_unet_device="cuda:0", manual_clip_device="cuda:1", manual_vae_device="cuda:1"):

        unet_dev, clip_dev, vae_dev = select_optimal_devices(
            mode=distribution_mode,
            manual_unet=manual_unet_device,
            manual_clip=manual_clip_device,
            manual_vae=manual_vae_device
        )

        logger.info(f"Universal Loader: Starting loading for {model_type} (UNet->{unet_dev}, CLIP->{clip_dev}, VAE->{vae_dev})...")

        # 1. Load UNet
        unet_path = resolve_unet_path(unet_name)
        if is_gguf_file(unet_path):
            dequant = weight_dtype if weight_dtype in ["float16", "bfloat16", "float32"] else "default"
            model = load_unet_gguf_on_device(unet_path, target_device=unet_dev, dequant_dtype=dequant)
        else:
            model_options = {
                "load_device": torch.device(unet_dev),
                "offload_device": mm.unet_offload_device()
            }
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
            elif weight_dtype == "float16":
                model_options["dtype"] = torch.float16
            elif weight_dtype == "bfloat16":
                model_options["dtype"] = torch.bfloat16
            elif weight_dtype == "float32":
                model_options["dtype"] = torch.float32
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        # 2. Load CLIP
        clip = None
        c_type = resolve_clip_type(model_type, unet_name)

        if clip_name1 != "none" and clip_name2 != "none":
            p1 = resolve_clip_path(clip_name1)
            p2 = resolve_clip_path(clip_name2)
            if is_gguf_file(p1) or is_gguf_file(p2):
                clip = load_clip_gguf_on_device([p1, p2], clip_type=c_type, target_device=clip_dev)
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=[p1, p2],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=c_type,
                    model_options={"load_device": torch.device(clip_dev), "offload_device": torch.device("cpu")}
                )
        elif clip_name1 != "none":
            p1 = resolve_clip_path(clip_name1)
            if is_gguf_file(p1):
                clip = load_clip_gguf_on_device([p1], clip_type=c_type, target_device=clip_dev)
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=[p1],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=c_type,
                    model_options={"load_device": torch.device(clip_dev), "offload_device": torch.device("cpu")}
                )

        # 3. Load VAE
        vae = None
        if vae_name != "none":
            vae_path = folder_paths.get_full_path("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        logger.info(f"✅ Universal Loader: Successfully loaded all components! (Resolved CLIPType: {c_type})")
        return (model, clip, vae)
