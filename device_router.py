import torch
import logging
from .device_manager import get_available_devices, get_gpu_status

logger = logging.getLogger("AutoMultiGPU")

class AutoDeviceRouter:
    """
    Intelligent Device Router:
    Outputs device strings or MULTIGPUDEVICE type based on real-time VRAM availability.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "routing_rule": ([
                    "Auto (Least Used GPU / Most Free VRAM)",
                    "Primary Compute GPU (cuda:0)",
                    "Secondary Offload GPU (cuda:1)",
                    "CPU Fallback"
                ], {"default": "Auto (Least Used GPU / Most Free VRAM)"}),
            }
        }

    RETURN_TYPES = ("MULTIGPUDEVICE", "STRING")
    RETURN_NAMES = ("multigpu_device", "device_str")
    FUNCTION = "route_device"
    CATEGORY = "AutoMultiGPU/utils"
    TITLE = "Auto Device Router (MultiGPU)"
    DESCRIPTION = "Dynamically inspects VRAM and returns the best device name for custom node inputs."

    def route_device(self, routing_rule="Auto (Least Used GPU / Most Free VRAM)"):
        devs = get_available_devices()
        gpu_stats = get_gpu_status()

        if routing_rule == "Primary Compute GPU (cuda:0)":
            chosen = "cuda:0" if "cuda:0" in devs else "cpu"
        elif routing_rule == "Secondary Offload GPU (cuda:1)":
            chosen = "cuda:1" if "cuda:1" in devs else ("cuda:0" if "cuda:0" in devs else "cpu")
        elif routing_rule == "CPU Fallback":
            chosen = "cpu"
        else:
            # Auto: Least used GPU (Most free VRAM)
            if gpu_stats:
                sorted_gpus = sorted(gpu_stats.keys(), key=lambda k: gpu_stats[k]["free_gb"], reverse=True)
                chosen = sorted_gpus[0]
            else:
                chosen = "cuda:0" if "cuda:0" in devs else "cpu"

        logger.info(f"Auto Device Router selected: {chosen}")
        return (chosen, chosen)
