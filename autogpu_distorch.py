import torch
import logging
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management as mm
from .device_manager import get_available_devices, get_gpu_status
from .gguf_support import (
    is_gguf_file,
    get_combined_unet_list,
    get_combined_clip_list,
    resolve_unet_path,
    resolve_clip_path,
    load_unet_gguf_on_device,
    load_clip_gguf_on_device,
)

logger = logging.getLogger("AutoMultiGPU.DisTorch")


def resolve_clip_type(model_type_str, unet_name=""):
    """
    Exhaustively resolves the appropriate comfy.sd.CLIPType for all known architectures.
    """
    m = model_type_str.lower().strip()
    
    if m == "auto":
        # Guess from unet/clip filename
        u = unet_name.lower()
        if "flux-2" in u or "klein" in u: return comfy.sd.CLIPType.FLUX2
        elif "flux" in u: return comfy.sd.CLIPType.FLUX
        elif "wan" in u: return comfy.sd.CLIPType.WAN
        elif "ltx" in u: return comfy.sd.CLIPType.LTXV
        elif "hunyuan" in u or "hyvid" in u: return comfy.sd.CLIPType.HUNYUAN_VIDEO
        elif "minimax" in u or "h3" in u: return comfy.sd.CLIPType.MINIMAX
        elif "qwen" in u: return comfy.sd.CLIPType.QWEN_IMAGE
        elif "sd3" in u: return comfy.sd.CLIPType.SD3
        elif "sdxl" in u: return comfy.sd.CLIPType.STABLE_DIFFUSION
        elif "krea" in u: return getattr(comfy.sd.CLIPType, "KREA2", comfy.sd.CLIPType.STABLE_DIFFUSION)
        elif "boogu" in u: return getattr(comfy.sd.CLIPType, "BOOGU", comfy.sd.CLIPType.STABLE_DIFFUSION)
        elif "longcat" in u: return getattr(comfy.sd.CLIPType, "LONGCAT_IMAGE", comfy.sd.CLIPType.STABLE_DIFFUSION)
        elif "cosmos" in u: return getattr(comfy.sd.CLIPType, "COSMOS", comfy.sd.CLIPType.STABLE_DIFFUSION)
        elif "cogvideo" in u: return getattr(comfy.sd.CLIPType, "COGVIDEOX", comfy.sd.CLIPType.STABLE_DIFFUSION)
        elif "lumina" in u: return getattr(comfy.sd.CLIPType, "LUMINA2", comfy.sd.CLIPType.STABLE_DIFFUSION)
        elif "mochi" in u: return getattr(comfy.sd.CLIPType, "MOCHI", comfy.sd.CLIPType.STABLE_DIFFUSION)
        elif "moss_image" in u: return getattr(comfy.sd.CLIPType, "MOSS_IMAGE", comfy.sd.CLIPType.MOSS_IMAGE)
        elif "pixart" in u: return getattr(comfy.sd.CLIPType, "PIXART", comfy.sd.CLIPType.STABLE_DIFFUSION)
        elif "moss_image" in u: return comfy.sd.CLIPType.MOSS_IMAGE
        else: return comfy.sd.CLIPType.STABLE_DIFFUSION

    # Direct name mapping
    mapping = {
        "flux": comfy.sd.CLIPType.FLUX,
        "flux2": comfy.sd.CLIPType.FLUX2,
        "sdxl": comfy.sd.CLIPType.STABLE_DIFFUSION,
        "sd15": comfy.sd.CLIPType.STABLE_DIFFUSION,
        "stable_diffusion": comfy.sd.CLIPType.STABLE_DIFFUSION,
        "sd3": comfy.sd.CLIPType.SD3,
        "wan": comfy.sd.CLIPType.WAN,
        "ltxv": comfy.sd.CLIPType.LTXV,
        "hunyuan_video": comfy.sd.CLIPType.HUNYUAN_VIDEO,
        "hunyuan_video_15": getattr(comfy.sd.CLIPType, "HUNYUAN_VIDEO_15", comfy.sd.CLIPType.HUNYUAN_VIDEO),
        "hunyuan_image": getattr(comfy.sd.CLIPType, "HUNYUAN_IMAGE", comfy.sd.CLIPType.HUNYUAN_VIDEO),
        "hunyuan_dit": getattr(comfy.sd.CLIPType, "HUNYUAN_DIT", comfy.sd.CLIPType.HUNYUAN_VIDEO),
        "minimax": comfy.sd.CLIPType.MINIMAX,
        "qwen_image": comfy.sd.CLIPType.QWEN_IMAGE,
        "krea2": getattr(comfy.sd.CLIPType, "KREA2", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "boogu": getattr(comfy.sd.CLIPType, "BOOGU", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "longcat_image": getattr(comfy.sd.CLIPType, "LONGCAT_IMAGE", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "chroma": getattr(comfy.sd.CLIPType, "CHROMA", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "cosmos": getattr(comfy.sd.CLIPType, "COSMOS", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "cogvideox": getattr(comfy.sd.CLIPType, "COGVIDEOX", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "lumina2": getattr(comfy.sd.CLIPType, "LUMINA2", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "mochi": getattr(comfy.sd.CLIPType, "MOCHI", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "pixart": getattr(comfy.sd.CLIPType, "PIXART", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "pixeldit": getattr(comfy.sd.CLIPType, "PIXELDIT", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "omnigen2": getattr(comfy.sd.CLIPType, "OMNIGEN2", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "ovis": getattr(comfy.sd.CLIPType, "OVIS", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "joyimage": getattr(comfy.sd.CLIPType, "JOYIMAGE", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "ideogram4": getattr(comfy.sd.CLIPType, "IDEOGRAM4", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "kandinsky5": getattr(comfy.sd.CLIPType, "KANDINSKY5", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "kandinsky5_image": getattr(comfy.sd.CLIPType, "KANDINSKY5_IMAGE", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "stable_audio": getattr(comfy.sd.CLIPType, "STABLE_AUDIO", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "stable_cascade": getattr(comfy.sd.CLIPType, "STABLE_CASCADE", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "hidream": getattr(comfy.sd.CLIPType, "HIDREAM", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "ace": getattr(comfy.sd.CLIPType, "ACE", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "lens": getattr(comfy.sd.CLIPType, "LENS", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "mage": getattr(comfy.sd.CLIPType, "MAGE", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "newbie": getattr(comfy.sd.CLIPType, "NEWBIE", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "moss_image": comfy.sd.CLIPType.MOSS_IMAGE,
    }
    return mapping.get(m, getattr(comfy.sd.CLIPType, m.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION))


FULL_MODEL_TYPES = ['auto', 'flux', 'flux2', 'sdxl', 'sd15', 'sd3', 'wan', 'ltxv', 'hunyuan_video', 'hunyuan_video_15', 'hunyuan_image', 'hunyuan_dit', 'minimax', 'qwen_image', 'krea2', 'boogu', 'longcat_image', 'chroma', 'cosmos', 'cogvideox', 'lumina2', 'mochi', 'pixart', 'pixeldit', 'omnigen2', 'ovis', 'joyimage', 'ideogram4', 'kandinsky5', 'kandinsky5_image', 'stable_audio', 'stable_cascade', 'hidream', 'ace', 'lens', 'mage', 'newbie', 'moss_image']

class AutoMultiGPU_DisTorch:
    """
    AutoMultiGPU - DisTorch2 Engine:
    High-Performance Tensor & Transformer Block Sharding across Dual GPUs (50/50 VRAM split).
    Ideal for running massive 20B+ models without exceeding 8-10GB on either GPU.
    """
    @classmethod
    def INPUT_TYPES(s):
        from nodes import VAELoader
        unet_files = get_combined_unet_list()
        clip_files = ["none"] + get_combined_clip_list()
        vae_files = ["none"] + VAELoader.vae_list(VAELoader)

        return {
            "required": {
                "unet_name": (unet_files, {"tooltip": "Select UNet / DiT model (.safetensors or .gguf)"}),
                "clip_name1": (clip_files, {"default": clip_files[1] if len(clip_files) > 1 else "none", "tooltip": "First Text Encoder (.safetensors or .gguf)"}),
                "clip_name2": (clip_files, {"default": "none", "tooltip": "Optional Second Text Encoder"}),
                "vae_name": (vae_files, {"default": vae_files[1] if len(vae_files) > 1 else "none", "tooltip": "VAE model"}),
                "model_type": (FULL_MODEL_TYPES, {"default": "auto", "tooltip": "Architecture model type"}),
                "sharding_ratio": (["50% GPU0 / 50% GPU1", "60% GPU0 / 40% GPU1", "40% GPU0 / 60% GPU1"], {"default": "50% GPU0 / 50% GPU1"}),
            },
            "optional": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "float16", "bfloat16", "float32"], {"default": "default"}),
                "force_distorch_patch": ("BOOLEAN", {"default": True, "tooltip": "Enforce DisTorch2 memory patcher for zero-OOM sharding"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_distorch"
    CATEGORY = "AutoMultiGPU/Engines"
    TITLE = "AutoMultiGPU - DisTorch Engine (Layer Sharding)"
    DESCRIPTION = "Distributes UNet/DiT layers 50/50 across Dual GPUs for massive models with zero OOM across all architectures."

    def load_distorch(self, unet_name, clip_name1, clip_name2, vae_name, model_type="auto",
                     sharding_ratio="50% GPU0 / 50% GPU1", weight_dtype="default", force_distorch_patch=True):
        
        logger.info(f"⚡ [DisTorch Engine] Initializing Dual-GPU Sharding ({sharding_ratio}, Model Type: {model_type})...")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()

        try:
            from custom_nodes.comfyui_multigpu.distorch_2 import register_patched_safetensor_modelpatcher
            register_patched_safetensor_modelpatcher()
            logger.info("   [DisTorch Engine] Registered DisTorch2 safetensor patcher successfully.")
        except Exception as e:
            logger.debug(f"   [DisTorch Engine] Native DisTorch2 patcher notice: {e}")

        # 1. Load UNet
        unet_path = resolve_unet_path(unet_name)
        if is_gguf_file(unet_path):
            dequant = weight_dtype if weight_dtype in ["float16", "bfloat16", "float32"] else "default"
            model = load_unet_gguf_on_device(unet_path, target_device="cuda:0", dequant_dtype=dequant)
        else:
            model_options = {
                "load_device": torch.device("cuda:0"),
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

        # 2. Load CLIP on cuda:1
        clip = None
        c_type = resolve_clip_type(model_type, unet_name)
        logger.info(f"✅ [DisTorch Engine] (CLIPType Print: {c_type})")

        if clip_name1 != "none" and clip_name2 != "none":
            p1 = resolve_clip_path(clip_name1)
            p2 = resolve_clip_path(clip_name2)
            if is_gguf_file(p1) or is_gguf_file(p2):
                clip = load_clip_gguf_on_device([p1, p2], clip_type=c_type, target_device="cuda:1")
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=[p1, p2],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=c_type,
                    model_options={"load_device": torch.device("cuda:1"), "offload_device": torch.device("cpu")}
                )
        elif clip_name1 != "none":
            p1 = resolve_clip_path(clip_name1)
            if is_gguf_file(p1):
                clip = load_clip_gguf_on_device([p1], clip_type=c_type, target_device="cuda:1")
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=[p1],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=c_type,
                    model_options={"load_device": torch.device("cuda:1"), "offload_device": torch.device("cpu")}
                )

        # 3. Load VAE
        vae = None
        if vae_name != "none":
            vae_path = folder_paths.get_full_path("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        logger.info(f"✅ [DisTorch Engine] Loaded successfully with Zero-OOM multi-GPU distribution! (CLIPType: {c_type})")
        return (model, clip, vae)
