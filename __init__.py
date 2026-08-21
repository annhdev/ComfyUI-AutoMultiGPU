import logging
from .device_manager import get_gpu_status, get_available_devices
from .checkpoint_loader import AutoCheckpointLoaderMultiGPU
from .universal_loader import AutoUniversalLoaderMultiGPU
from .modular_loaders import (
    AutoUNETLoaderMultiGPU,
    AutoDualCLIPLoaderMultiGPU,
    AutoCLIPLoaderMultiGPU,
    AutoVAELoaderMultiGPU,
)
from .minimax_h3_decoder import AutoMiniMaxH3VAEDecodeMultiGPU
from .device_router import AutoDeviceRouter
from .autogpu_distorch import AutoMultiGPU_DisTorch
from .autogpu_pytorch import AutoMultiGPU_PyTorch
from .autogpu_automatic import AutoMultiGPU_Automatic

logger = logging.getLogger("AutoMultiGPU")

NODE_CLASS_MAPPINGS = {
    # The 3 Core Production Engines
    "AutoMultiGPU_Automatic": AutoMultiGPU_Automatic,
    "AutoMultiGPU_DisTorch": AutoMultiGPU_DisTorch,
    "AutoMultiGPU_PyTorch": AutoMultiGPU_PyTorch,
    
    # Universal & Modular Loaders
    "AutoUniversalLoaderMultiGPU": AutoUniversalLoaderMultiGPU,
    "AutoCheckpointLoaderMultiGPU": AutoCheckpointLoaderMultiGPU,
    "AutoUNETLoaderMultiGPU": AutoUNETLoaderMultiGPU,
    "AutoDualCLIPLoaderMultiGPU": AutoDualCLIPLoaderMultiGPU,
    "AutoCLIPLoaderMultiGPU": AutoCLIPLoaderMultiGPU,
    "AutoVAELoaderMultiGPU": AutoVAELoaderMultiGPU,
    "AutoMiniMaxH3VAEDecodeMultiGPU": AutoMiniMaxH3VAEDecodeMultiGPU,
    "AutoDeviceRouter": AutoDeviceRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMultiGPU_Automatic": "AutoMultiGPU - Automatic Engine (Zero-OOM Governor)",
    "AutoMultiGPU_DisTorch": "AutoMultiGPU - DisTorch Engine (Layer Sharding)",
    "AutoMultiGPU_PyTorch": "AutoMultiGPU - PyTorch Engine (Native Device Split)",
    "AutoUniversalLoaderMultiGPU": "Auto Universal Loader (MultiGPU - Safetensors & GGUF)",
    "AutoCheckpointLoaderMultiGPU": "Auto Checkpoint Loader (MultiGPU)",
    "AutoUNETLoaderMultiGPU": "Auto UNET Loader (MultiGPU - Safetensors & GGUF)",
    "AutoDualCLIPLoaderMultiGPU": "Auto DualCLIP Loader (MultiGPU - Safetensors & GGUF)",
    "AutoCLIPLoaderMultiGPU": "Auto CLIP Loader (MultiGPU - Safetensors & GGUF)",
    "AutoVAELoaderMultiGPU": "Auto VAE Loader (MultiGPU)",
    "AutoMiniMaxH3VAEDecodeMultiGPU": "Auto MiniMax H3 VAE Decode (Zero-OOM)",
    "AutoDeviceRouter": "Auto Device Router (MultiGPU)",
}

logger.info("==================================================")
logger.info("[AutoMultiGPU] ComfyUI-AutoMultiGPU loaded with 3 Production Engines:")
logger.info("   1. AutoMultiGPU - DisTorch Engine (Layer Sharding)")
logger.info("   2. AutoMultiGPU - PyTorch Engine (Native Device Split)")
logger.info("   3. AutoMultiGPU - Automatic Engine (Zero-OOM Governor)")
gpus = get_gpu_status()
if gpus:
    for dev, info in gpus.items():
        logger.info(f"   Detected {dev}: {info['name']} | Total: {info['total_gb']} GB | Free: {info['free_gb']} GB")
logger.info("==================================================")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
