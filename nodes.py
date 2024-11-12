import os
from comfy.utils import ProgressBar
import folder_paths

from .core.test_xportrait import main as xportrait_main

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
        
    @classmethod
    def INPUT_TYPES(s):
        return {}
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    DESCRIPTION = "Downloads and loads the X-Portrait model"
      
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
        
        