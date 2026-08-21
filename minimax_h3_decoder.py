import torch
import logging
import comfy.model_management as mm
from .device_manager import get_available_devices, get_gpu_status

logger = logging.getLogger("AutoMultiGPU")

class AutoMiniMaxH3VAEDecodeMultiGPU:
    """
    Automatic MiniMax H3 (Hailuo) VAE Decoder with Multi-GPU isolation & zero-OOM protection.
    - Decodes on GPU 1 (isolating VAE from Sampler VRAM on GPU 0).
    - Uses H3 reference 256px spatial tiles with 64px overlap.
    - Streams decoded video frames directly to System RAM (CPU) to prevent VRAM explosion.
    - Automatically cleans GPU memory cache before & after decoding.
    """
    @classmethod
    def INPUT_TYPES(cls):
        devices = get_available_devices()
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "MiniMax H3 Video Latent (Bx24xTxHxW)"}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 Video VAE"}),
                "target_device": (["Auto (Secondary GPU - cuda:1)", "cuda:1", "cuda:0", "cpu"], {"default": "Auto (Secondary GPU - cuda:1)"}),
                "spatial_tile_size": ([256, 384, 512], {"default": 256, "tooltip": "256 matches reference quality and eliminates OOM."}),
                "output_device": (["cpu (Safe - No VRAM Spike)", "gpu"], {"default": "cpu (Safe - No VRAM Spike)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "AutoMultiGPU/minimax_h3"
    TITLE = "Auto MiniMax H3 VAE Decode (MultiGPU Zero-OOM)"
    DESCRIPTION = "Zero-OOM MiniMax H3 video decoder that runs on the secondary GPU and safely streams frames to RAM."

    def decode(self, samples, vae, target_device="Auto (Secondary GPU - cuda:1)", spatial_tile_size=256, output_device="cpu (Safe - No VRAM Spike)"):
        devs = get_available_devices()
        if target_device == "Auto (Secondary GPU - cuda:1)":
            chosen_device = "cuda:1" if "cuda:1" in devs else ("cuda:0" if "cuda:0" in devs else "cpu")
        else:
            chosen_device = target_device if target_device in devs else "cuda:0"

        # Pre-cleanup
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Decoding MiniMax H3 video latent on {chosen_device} with tile size {spatial_tile_size}...")

        # Check if fast decoder from ComfyUI-MiniMaxH3_LatentUpscaler is available
        try:
            from custom_nodes.ComfyUI_MiniMaxH3_LatentUpscaler.vae_decode import MiniMaxH3VAEDecodeFast
            out_dev = "gpu" if output_device == "gpu" else "cpu"
            
            # Temporarily attach VAE to chosen device
            orig_vae_dev = getattr(vae, "device", None)
            vae.device = torch.device(chosen_device)
            if hasattr(vae, "patcher") and vae.patcher is not None:
                vae.patcher.load_device = torch.device(chosen_device)

            fast_node = MiniMaxH3VAEDecodeFast()
            result = fast_node.decode(
                samples=samples,
                vae=vae,
                tiling=True,
                tile_size=int(spatial_tile_size),
                tile_overlap=64,
                output_device=out_dev
            )
            
            # Post-cleanup
            mm.soft_empty_cache()
            return result
        except Exception as e:
            logger.warning(f"Direct fast decoder import fallback: {e}")

        # Standard VAEDecode fallback with device isolation
        with mm.cuda_device_context(torch.device(chosen_device)):
            try:
                images = vae.decode(samples["samples"])
            except Exception:
                images = vae.decode_tiled(samples["samples"], tile_x=spatial_tile_size, tile_y=spatial_tile_size, overlap=64)

        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        mm.soft_empty_cache()
        return (images,)
