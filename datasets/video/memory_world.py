from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import torch
import random
from omegaconf import DictConfig
from torchvision.datasets.video_utils import _VideoTimestampsDataset, _collate_fn
from tqdm import tqdm
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)
from .utils import read_video

def _parse_camera_matrix(data):
    # data is list of 16 floats
    return torch.tensor(data, dtype=torch.float32).view(4, 4)

class MemoryWorldBaseVideoDataset(BaseVideoDataset):
    def _should_download(self) -> bool:
        return False

    def download_dataset(self) -> None:
        pass

    def build_metadata(self, split: SPLIT) -> None:
        if (self.metadata_dir / f"{split}.pt").exists():
            return

        # Scan all mp4 files recursively
        video_paths = sorted(list(self.save_dir.glob("**/*.mp4")), key=str)
        # Exclude metadata/latents folders if they exist inside save_dir
        video_paths = [p for p in video_paths if "metadata" not in p.parts and "latents" not in p.name]

        # Deterministic 90/10 split
        rng = random.Random(42)
        rng.shuffle(video_paths)
        
        n = len(video_paths)
        n_train = int(n * 0.9)
        
        splits = {
            "training": video_paths[:n_train],
            "validation": video_paths[n_train:],
            "test": video_paths[n_train:]
        }
        
        target_paths = splits.get(split, [])
        if not target_paths:
            print(f"Warning: No videos found for split {split}")
            return

        dl = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(target_paths),
            batch_size=16,
            num_workers=16,
            collate_fn=_collate_fn,
        )
        
        video_pts = []
        video_fps = []
        
        with tqdm(total=len(dl), desc=f"Building metadata for {split}") as pbar:
            for batch in dl:
                pbar.update(1)
                batch_pts, batch_fps = list(zip(*batch))
                batch_pts = [torch.as_tensor(pts, dtype=torch.long) for pts in batch_pts]
                video_pts.extend(batch_pts)
                video_fps.extend(batch_fps)
                
        metadata = {
            "video_paths": target_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        video_path = video_metadata["video_paths"]
        # datasets.video.utils.read_video returns only video frames.
        vframes = read_video(str(video_path), pts_unit="sec")
        
        # Handle channel dimension if needed
        if vframes.shape[-1] == 3:
            vframes = vframes.permute(0, 3, 1, 2)
            
        T = vframes.shape[0]
        # User logic: replace frame 0 with frame 1
        if T > 1:
            vframes[0] = vframes[1]
            
        # User logic: pad to 154 frames (77 context + 77 predict)
        target_total = 154
        if T < target_total:
            padding = vframes[-1:].repeat(target_total - T, 1, 1, 1)
            vframes = torch.cat([vframes, padding], dim=0)
            
        vframes = vframes[:target_total]
        vframes = vframes.float() / 255.0
        
        if end_frame is None:
            end_frame = target_total
            
        return vframes[start_frame:end_frame]

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        video_path = Path(video_metadata["video_paths"])
        parent_dir = video_path.parent
        
        # Search for corresponding json file
        target_json = None
        for f in parent_dir.iterdir():
            if f.suffix == '.json' and '_character.json' not in f.name and '_check.json' not in f.name:
                target_json = f
                break
        
        raw_matrices = []
        if target_json:
            try:
                with open(target_json, 'r') as f:
                    data = json.load(f)
                for i in range(153):
                    key = str(i)
                    if key in data:
                        mat = _parse_camera_matrix(data[key])
                    else:
                        mat = raw_matrices[-1] if raw_matrices else torch.eye(4)
                    raw_matrices.append(mat)
            except Exception as e:
                print(f"Error reading camera json {target_json}: {e}")
                raw_matrices = [torch.eye(4)] * 153
        else:
            raw_matrices = [torch.eye(4)] * 153
            
        # User logic: replace 0 with 1
        if len(raw_matrices) > 1:
            raw_matrices[0] = raw_matrices[1]
            
        # Append last to reach 154
        raw_matrices.append(raw_matrices[-1])
        
        matrices = torch.stack(raw_matrices).view(154, -1) # (154, 16)
        
        return matrices[start_frame:end_frame]

class MemoryWorldSimpleVideoDataset(MemoryWorldBaseVideoDataset, BaseSimpleVideoDataset):
    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimpleVideoDataset.__init__(self, cfg, split)

class MemoryWorldAdvancedVideoDataset(MemoryWorldBaseVideoDataset, BaseAdvancedVideoDataset):
    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)
