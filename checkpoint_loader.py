import torch
import logging
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management as mm
from .device_manager import get_available_devices, select_optimal_devices, get_gpu_status

logger = logging.getLogger("AutoMultiGPU")

class AutoCheckpointLoaderMultiGPU:
    """
    Pure PyTorch Multi-GPU Checkpoint Loader.
    Zero monkey-patching: passes native PyTorch load_device options to ComfyUI core.
    - MODEL (UNet/DiT) -> Primary Compute GPU (cuda:0)
    - CLIP (Text Encoder) -> Secondary GPU (cuda:1)
    - VAE -> Secondary GPU (cuda:1)
    """
    @classmethod
    def INPUT_TYPES(s):
        devices = get_available_devices()
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The checkpoint to load."}),
                "distribution_mode": ([
                    "Auto: UNet GPU0 / Encoders GPU1",
                    "Auto: Balanced Free VRAM",
                    "Auto: Split Video (GPU0 Sampler / GPU1 VAE)",
                    "Manual Override"
                ], {"default": "Auto: UNet GPU0 / Encoders GPU1"}),
            },
            "optional": {
                "manual_unet_device": (devices, {"default": "cuda:0" if "cuda:0" in devices else "cpu"}),
                "manual_clip_device": (devices, {"default": "cuda:1" if "cuda:1" in devices else "cpu"}),
                "manual_vae_device": (devices, {"default": "cuda:1" if "cuda:1" in devices else "cpu"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = (
        "Model (UNet/DiT) routed to Primary/Compute GPU.",
        "CLIP (Text Encoder) routed to Secondary GPU.",
        "VAE (Encode/Decode) routed to Secondary GPU."
    )
    FUNCTION = "load_checkpoint"
    CATEGORY = "AutoMultiGPU"
    TITLE = "Auto Checkpoint Loader (MultiGPU - Pure PyTorch)"
    DESCRIPTION = "Pure PyTorch native checkpoint loader with automatic multi-GPU distribution."

    def load_checkpoint(self, ckpt_name, distribution_mode="Auto: UNet GPU0 / Encoders GPU1",
                        manual_unet_device="cuda:0", manual_clip_device="cuda:1", manual_vae_device="cuda:1"):
        
        unet_dev, clip_dev, vae_dev = select_optimal_devices(
            mode=distribution_mode,
            manual_unet=manual_unet_device,
            manual_clip=manual_clip_device,
            manual_vae=manual_vae_device
        )

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)

        model_options = {
            "load_device": torch.device(unet_dev),
            "offload_device": mm.unet_offload_device()
        }

        te_model_options = {
            "load_device": torch.device(clip_dev),
            "offload_device": mm.unet_offload_device()
        }

        logger.info(f"Loading checkpoint [{ckpt_name}] via Pure PyTorch:\n  MODEL -> {unet_dev} | CLIP -> {clip_dev} | VAE -> {vae_dev}")

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            output_clipvision=False,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            output_model=True,
            model_options=model_options,
            te_model_options=te_model_options
        )

        model, clip, vae = out[0], out[1], out[2]

        if vae is not None:
            try:
                vae.device = torch.device(vae_dev)
                if hasattr(vae, "patcher") and vae.patcher is not None:
                    vae.patcher.load_device = torch.device(vae_dev)
            except Exception as e:
                logger.debug(f"VAE device note: {e}")

        return (model, clip, vae)
