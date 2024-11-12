import os
from comfy.utils import ProgressBar
import folder_paths
import torch

from .model_lib.ControlNet.cldm.model import create_model
from .core.test_xportrait import main as xportrait_main

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class BaseNode:
    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self, *args, **kwargs):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None
        
    CATEGORY = "X-Portrait"
    
class DownloadXPortraitModel(BaseNode):
    def __init__(self):
        super().__init__()
        self.model_config = "config/cldm_v15_appearance_pose_local_mm.yaml"
        
    @classmethod
    def INPUT_TYPES(s):
        return {}
    
    RETURN_TYPES = ("XPORTRAIT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    DESCRIPTION = "Downloads and loads the X-Portrait model"
    
    def load_state_dict(self, model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
        print(f"Loading model state dict from {ckpt_path} ...")
        state_dict = torch.load(ckpt_path, map_location=map_location)
        state_dict = state_dict.get('state_dict', state_dict)

        if reinit_hint_block:
            print("Ignoring hint block parameters from checkpoint!")
            for k in list(state_dict.keys()):
                if k.startswith("control_model.input_hint_block"):
                    state_dict.pop(k)

        # Load the state dict
        model.load_state_dict(state_dict, strict=strict)
        del state_dict
      
    def load_model(self):
        model_dir = os.path.join(folder_paths.models_dir, "x-portrait")
        model_file = "model_state-415001.th"
        model_path = os.path.join(model_dir, model_file)
        
        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_dir}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="fffiloni/X-Portrait",
                            allow_patterns=[f"*model_state-415001.th*"],
                            local_dir=model_dir,
                            local_dir_use_symlinks=False)
            
        print(f"Model loaded from: {model_path}")
        
        # ******************************
        # create model
        # ******************************
        model = create_model(self.model_config).cpu()
        model.sd_locked = True
        model.only_mid_control = False

        # ******************************
        # load pre-trained models
        # ******************************
        if model_path is not None:
            self.load_state_dict(model, model_path, strict=False)
        else:
            print('please privide the correct resume_dir!')
            exit()
        
        infer_model = model.module if hasattr(model, "module") else model
        
        return (infer_model,)
    
# python3 core/test_xportrait.py \
# --model_config config/cldm_v15_appearance_pose_local_mm_copy.yaml \
# --output_dir outputs \
# --resume_dir checkpoint/model_state-415001.th \
# --seed 999 \
# --uc_scale 5 \
# --source_image assets/source_image.png \
# --driving_video assets/driving_video.mp4 \
# --best_frame 36 \
# --out_frames -1 \
# --use_fp16 \
# --num_mix 4 \
# --ddim_steps 30 \
# --num_drivings 16 \
    
    
class XPortrait(BaseNode):
    def __init__(self):
        super().__init__()
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "xportrait_model": ("XPORTRAIT_MODEL", {"forceInput": True}),
                "source_image": ("IMAGE",),
                "driving_video": ("IMAGE",),
                "seed": ("INT", {"default": 999}),
                "ddim_steps": ("INT", {"default": 15}),
                "cfg_scale": ("FLOAT", {"default": 5.0}),
                "best_frame": ("INT", {"default": 36}),
                "context_window": ("INT", {"default": 16}),
                "overlap": ("INT", {"default": 4}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    DESCRIPTION = "Generates a video from the X-Portrait model"
    
    def generate(self, xportrait_model, source_image, driving_video, seed, ddim_steps, cfg_scale, best_frame, context_window, overlap):
        args = Args(
            seed=seed,
            uc_scale=cfg_scale,
            source_image=source_image,
            driving_video=driving_video,
            best_frame=best_frame,
            out_frames=-1,
            num_drivings=context_window,
            num_mix=overlap,
            ddim_steps=ddim_steps,
        )
        output_videos = xportrait_main(args, xportrait_model)
        return (output_videos[0],)