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

class AutoUNETLoaderMultiGPU:
    """
    Universal Multi-GPU UNet / DiT Model Loader.
    Supports BOTH .safetensors and .gguf formats natively.
    Routes the main model to the primary compute GPU (default cuda:0) with dynamic CPU offload.
    """
    @classmethod
    def INPUT_TYPES(s):
        devices = get_available_devices()
        return {
            "required": {
                "unet_name": (get_combined_unet_list(), {"tooltip": "Select .safetensors or .gguf diffusion model."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "float16", "bfloat16", "float32"], {"default": "default"}),
                "target_device": (["Auto (Highest Free VRAM)", "cuda:0", "cuda:1"] + [d for d in devices if d not in ["cuda:0", "cuda:1"]], {"default": "Auto (Highest Free VRAM)"}),
                "cpu_offload": (["auto_smart_offload", "always_offload_to_cpu", "disabled_vram_only"], {"default": "auto_smart_offload", "tooltip": "Automatically offload weights to CPU RAM when VRAM is constrained to prevent OOM."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "AutoMultiGPU/modular"
    TITLE = "Auto UNET Loader (MultiGPU - Safetensors & GGUF)"
    DESCRIPTION = "Universal loader for .safetensors and .gguf diffusion models with smart GPU balancing and dynamic CPU offloading."

    def load_unet(self, unet_name, weight_dtype="default", target_device="Auto (Highest Free VRAM)", cpu_offload="auto_smart_offload"):
        devs = get_available_devices()
        gpu_stats = get_gpu_status()

        if target_device == "Auto (Highest Free VRAM)" and gpu_stats:
            sorted_gpus = sorted(gpu_stats.keys(), key=lambda k: gpu_stats[k]["free_gb"], reverse=True)
            chosen_device = sorted_gpus[0]
        elif target_device in devs:
            chosen_device = target_device
        else:
            chosen_device = "cuda:0" if "cuda:0" in devs else "cpu"

        use_cpu_offload = cpu_offload in ["auto_smart_offload", "always_offload_to_cpu"]

        unet_path = resolve_unet_path(unet_name)
        logger.info(f"Loading UNET [{unet_name}] (Format: {'GGUF' if is_gguf_file(unet_name) else 'Safetensors'}) on target device: {chosen_device} (CPU Offload: {cpu_offload})")

        # Check if file is GGUF format
        if is_gguf_file(unet_path):
            dequant = "default"
            if weight_dtype in ["float16", "bfloat16", "float32"]:
                dequant = weight_dtype
            model = load_unet_gguf_on_device(
                unet_path=unet_path,
                target_device=chosen_device,
                dequant_dtype=dequant,
                cpu_offload=use_cpu_offload
            )
            return (model,)

        # Standard Safetensors / PyTorch format
        model_options = {}
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

        dev_obj = torch.device(chosen_device)
        model_options["load_device"] = dev_obj
        model_options["offload_device"] = torch.device("cpu") if use_cpu_offload else mm.unet_offload_device()

        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)


class AutoDualCLIPLoaderMultiGPU:
    """
    Universal Multi-GPU Dual CLIP Loader.
    Supports BOTH .safetensors and .gguf text encoders across all 37 model architectures.
    Routes text encoders to the secondary GPU (default cuda:1) to keep GPU0 free for latents.
    """
    @classmethod
    def INPUT_TYPES(s):
        devices = get_available_devices()
        clip_list = get_combined_clip_list()
        return {
            "required": {
                "clip_name1": (clip_list, {"tooltip": "First CLIP / Text Encoder (.safetensors or .gguf)"}),
                "clip_name2": (clip_list, {"tooltip": "Second CLIP / Text Encoder (.safetensors or .gguf)"}),
                "type": (FULL_MODEL_TYPES, {"default": "auto", "tooltip": "Comprehensive Architecture Model Type"}),
                "target_device": (["Auto (Secondary GPU - cuda:1)", "cuda:1", "cuda:0", "cpu"], {"default": "Auto (Secondary GPU - cuda:1)"}),
                "cpu_offload": (["auto_smart_offload", "always_offload_to_cpu", "disabled_vram_only"], {"default": "auto_smart_offload"}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "AutoMultiGPU/modular"
    TITLE = "Auto DualCLIP Loader (MultiGPU - Safetensors & GGUF)"
    DESCRIPTION = "Universal Dual CLIP loader (.safetensors & .gguf) with secondary GPU offloading across all 37 model types."

    def load_clip(self, clip_name1, clip_name2, type="auto", target_device="Auto (Secondary GPU - cuda:1)", cpu_offload="auto_smart_offload"):
        devs = get_available_devices()
        if target_device == "Auto (Secondary GPU - cuda:1)":
            chosen_device = "cuda:1" if "cuda:1" in devs else ("cuda:0" if "cuda:0" in devs else "cpu")
        else:
            chosen_device = target_device if target_device in devs else "cuda:0"

        use_cpu_offload = cpu_offload in ["auto_smart_offload", "always_offload_to_cpu"]
        logger.info(f"Loading Dual CLIP on target device: {chosen_device} (CPU Offload: {cpu_offload}, Model Type: {type})")

        clip_type = resolve_clip_type(type, clip_name1)
        clip_path1 = resolve_clip_path(clip_name1)
        clip_path2 = resolve_clip_path(clip_name2)

        # Check if any is GGUF
        if is_gguf_file(clip_path1) or is_gguf_file(clip_path2):
            clip = load_clip_gguf_on_device(
                clip_paths=[clip_path1, clip_path2],
                clip_type=clip_type,
                target_device=chosen_device,
                cpu_offload=use_cpu_offload
            )
            return (clip,)

        model_options = {
            "load_device": torch.device(chosen_device),
            "offload_device": torch.device("cpu") if use_cpu_offload else mm.unet_offload_device()
        }

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )
        return (clip,)


class AutoCLIPLoaderMultiGPU:
    """
    Universal Multi-GPU Single CLIP Loader.
    Supports BOTH .safetensors and .gguf text encoders across all 37 model architectures.
    """
    @classmethod
    def INPUT_TYPES(s):
        devices = get_available_devices()
        return {
            "required": {
                "clip_name": (get_combined_clip_list(), {"tooltip": "Select .safetensors or .gguf text encoder."}),
                "type": (FULL_MODEL_TYPES, {"default": "auto", "tooltip": "Comprehensive Architecture Model Type"}),
                "target_device": (["Auto (Secondary GPU - cuda:1)", "cuda:1", "cuda:0", "cpu"], {"default": "Auto (Secondary GPU - cuda:1)"}),
                "cpu_offload": (["auto_smart_offload", "always_offload_to_cpu", "disabled_vram_only"], {"default": "auto_smart_offload"}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "AutoMultiGPU/modular"
    TITLE = "Auto CLIP Loader (MultiGPU - Safetensors & GGUF)"
    DESCRIPTION = "Universal Single CLIP loader (.safetensors & .gguf) with secondary GPU offloading across all 37 model types."

    def load_clip(self, clip_name, type="auto", target_device="Auto (Secondary GPU - cuda:1)", cpu_offload="auto_smart_offload"):
        devs = get_available_devices()
        if target_device == "Auto (Secondary GPU - cuda:1)":
            chosen_device = "cuda:1" if "cuda:1" in devs else ("cuda:0" if "cuda:0" in devs else "cpu")
        else:
            chosen_device = target_device if target_device in devs else "cuda:0"

        use_cpu_offload = cpu_offload in ["auto_smart_offload", "always_offload_to_cpu"]
        clip_path = resolve_clip_path(clip_name)
        logger.info(f"Loading CLIP [{clip_name}] on target device: {chosen_device} (CPU Offload: {cpu_offload}, Model Type: {type})")

        clip_type = resolve_clip_type(type, clip_name)

        if is_gguf_file(clip_path):
            clip = load_clip_gguf_on_device(
                clip_paths=[clip_path],
                clip_type=clip_type,
                target_device=chosen_device,
                cpu_offload=use_cpu_offload
            )
            return (clip,)

        model_options = {
            "load_device": torch.device(chosen_device),
            "offload_device": torch.device("cpu") if use_cpu_offload else mm.unet_offload_device()
        }

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )
        return (clip,)


class AutoVAELoaderMultiGPU:
    """
    Automatic Multi-GPU VAE Loader.
    Routes VAE to cuda:1 to eliminate memory competition with the UNet during generation.
    """
    @classmethod
    def INPUT_TYPES(s):
        from nodes import VAELoader
        devices = get_available_devices()
        return {
            "required": {
                "vae_name": (VAELoader.vae_list(VAELoader), ),
                "target_device": (["Auto (Secondary GPU - cuda:1)", "cuda:1", "cuda:0", "cpu"], {"default": "Auto (Secondary GPU - cuda:1)"}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "AutoMultiGPU/modular"
    TITLE = "Auto VAE Loader (MultiGPU)"
    DESCRIPTION = "Automatic Multi-GPU VAE Loader with secondary GPU offloading."

    def load_vae(self, vae_name, target_device="Auto (Secondary GPU - cuda:1)"):
        from nodes import VAELoader
        devs = get_available_devices()
        if target_device == "Auto (Secondary GPU - cuda:1)":
            chosen_device = "cuda:1" if "cuda:1" in devs else ("cuda:0" if "cuda:0" in devs else "cpu")
        else:
            chosen_device = target_device if target_device in devs else "cuda:0"

        logger.info(f"Loading VAE [{vae_name}] on target device: {chosen_device}")

        vae_tuple = VAELoader().load_vae(vae_name)
        vae = vae_tuple[0]
        
        try:
            vae.device = torch.device(chosen_device)
            if hasattr(vae, "patcher") and vae.patcher is not None:
                vae.patcher.load_device = torch.device(chosen_device)
        except Exception as e:
            logger.debug(f"VAE device assignment note: {e}")

        return (vae,)
