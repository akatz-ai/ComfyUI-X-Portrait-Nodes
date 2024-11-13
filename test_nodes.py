import argparse
import os
import numpy as np
import torch

import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_file_dir))

from model_lib.ControlNet.cldm.model import create_model
from core.test_xportrait import main as xportrait_main

from PIL import Image
from decord import VideoReader
from decord import cpu, gpu

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
class DownloadXPortraitModel():
    def __init__(self):
        super().__init__()
        self.model_config = os.path.join(current_file_dir, "config", "cldm_v15_appearance_pose_local_mm.yaml")
    
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
      
    def load_model(self, model_dir=None):
        model_file = "model_state-415001.th"
        model_path = os.path.join(model_dir, model_file)
        
        print(f"Model path: {model_path}")
        
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
    
    
class XPortrait():
    def __init__(self):
        super().__init__()
    
    def generate(self, xportrait_model, source_image, driving_video, seed=999, ddim_steps=5, cfg_scale=5.0, best_frame=36, context_window=16, overlap=4, fps=15):
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
            fps=fps,
            start_idx=0,
            skip=1,
            use_fp16=True,
            output_dir="outputs",
            control_type="appearance_pose_local_mm",
            control_mode="controlnet_important",
            wonoise=True,
            only_mid_control=False,
            sd_locked=True,
            reinit_hint_block=False,
            local_rank=0,
            device="cuda",
            eta=0.0,
            world_size=1,
            rank=0,
            ema_rate=0,
        )
        output_videos = xportrait_main(args, xportrait_model)
        print(f"Output videos: {output_videos}")
        print(f"Output videos type: {type(output_videos)}")
        return (output_videos[0],)
    
if __name__ == "__main__":
    # Get path to models folder from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--source-image-path", type=str, default=None)
    parser.add_argument("--driving-video-path", type=str, default=None)
    args = parser.parse_args()

    # Load image and video from assets files into tensors with shape [B, H, W, C]
    source_image = Image.open(args.source_image_path)

    # Convert image to a tensor with shape [B, H, W, C] of dtype float32 and values from 0 to 1
    source_image = torch.from_numpy(np.array(source_image)).float() / 255.0
    source_image = source_image.unsqueeze(0)  # Add batch dimension
    print(f"Source image shape: {source_image.shape}")
    print(f"Source image data peek: {source_image.view(-1)[:10]}")  # Flatten for data peek

    # Read video frames and convert to a tensor
    video_reader = VideoReader(args.driving_video_path, ctx=cpu(0))
    
    # Assuming VideoReader provides an iterable of frames
    frames = []
    for i in range(len(video_reader)):
        frame = video_reader[i].asnumpy()
        # frames are in [H, W, C] format and values from 0 to 255, we need to convert to [B, H, W, C]
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.unsqueeze(0)  # Add batch dimension
        frames.append(frame)

    driving_video = torch.cat(frames, dim=0)
    print(f"Driving video shape: {driving_video.shape}")

    xportrait_model = DownloadXPortraitModel().load_model(model_dir=args.models_dir)
    xportrait = XPortrait()
    xportrait.generate(xportrait_model, source_image, driving_video)