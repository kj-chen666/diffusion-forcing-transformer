from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re
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

def _parse_camera_matrix(data: Any) -> torch.Tensor:
    if isinstance(data, list) and len(data) == 16:
        return torch.tensor(data, dtype=torch.float32).view(4, 4)
    if isinstance(data, str):
        matches = re.findall(r"\[(.*?)\]", data)
        if len(matches) == 4:
            cols = []
            for m in matches:
                vals = [float(x) for x in m.strip().split()]
                if len(vals) != 4:
                    return torch.eye(4)
                cols.append(vals)
            return torch.tensor(cols, dtype=torch.float32).T
    return torch.eye(4)


def _parse_caption_line(line: str) -> Optional[str]:
    """
    Parse lines like:
      subfolder/video.mp4 <tab> caption
      subfolder/video.mp4    caption
    Returns video path string or None.
    """
    line = line.strip()
    if not line:
        return None
    if line.endswith(".mp4"):
        # Allow path-only lines.
        return line
    parts = re.split(r"\t+", line, maxsplit=1)
    if len(parts) < 2:
        parts = re.split(r"\s{2,}", line, maxsplit=1)
    if len(parts) != 2:
        return None
    video_path = parts[0].strip()
    if not video_path.endswith(".mp4"):
        return None
    return video_path

class MemoryWorldBaseVideoDataset(BaseVideoDataset):
    def _should_download(self) -> bool:
        return False

    def download_dataset(self) -> None:
        pass

    def _collect_video_paths(self) -> List[Path]:
        """
        Prefer samples listed in captions.txt to align with preprocessing pipeline.
        Fallback to recursive mp4 scan if no caption file is provided/found.
        """
        captions_path = self.cfg.get("captions_path", None)
        if captions_path is None:
            captions_path = self.save_dir / "captions.txt"
        else:
            captions_path = Path(captions_path)
        captions_root = Path(self.cfg.get("captions_root", self.save_dir))

        video_paths: List[Path] = []
        if captions_path.exists():
            with open(captions_path, "r", encoding="utf-8") as f:
                for line in f:
                    parsed = _parse_caption_line(line)
                    if parsed is None:
                        continue
                    p = Path(parsed)
                    p = p if p.is_absolute() else captions_root / p
                    if p.exists():
                        video_paths.append(p.resolve())
            if len(video_paths) > 0:
                # de-duplicate while preserving original captions order
                return list(dict.fromkeys(video_paths))

        # Fallback: scan all mp4 recursively.
        video_paths = sorted(list(self.save_dir.glob("**/*.mp4")), key=str)
        return [
            p
            for p in video_paths
            if "metadata" not in p.parts and "latents" not in p.name
        ]

    def build_metadata(self, split: SPLIT) -> None:
        if (self.metadata_dir / f"{split}.pt").exists():
            return

        video_paths = self._collect_video_paths()
        if not video_paths:
            raise RuntimeError(
                f"No videos found in save_dir={self.save_dir}. "
                "Please check dataset.save_dir / dataset.captions_path."
            )

        # Deterministic split from captions/video list.
        rng = random.Random(int(self.cfg.get("split_seed", 42)))
        rng.shuffle(video_paths)
        n = len(video_paths)
        n_train = int(n * float(self.cfg.get("train_ratio", 0.9)))

        splits = {
            "training": video_paths[:n_train],
            "validation": video_paths[n_train:],
            "test": video_paths[n_train:],
        }
        target_paths = splits.get(split, [])
        if not target_paths:
            raise RuntimeError(f"No videos found for split {split}")

        dl = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(target_paths),
            batch_size=16,
            num_workers=int(self.cfg.get("metadata_num_workers", 16)),
            collate_fn=_collate_fn,
        )
        video_pts = []
        video_fps = []

        with tqdm(total=len(dl), desc=f"Building metadata for {split}") as pbar:
            for batch in dl:
                pbar.update(1)
                batch_pts, batch_fps = list(zip(*batch))
                batch_pts = [
                    torch.as_tensor(pts, dtype=torch.long) for pts in batch_pts
                ]
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

        if vframes.shape[0] == 0:
            raise RuntimeError(f"Failed to decode video: {video_path}")

        T = vframes.shape[0]
        # Align with vae_process_dataset.py:
        # 1) replace frame 0 with frame 1
        if T > 1:
            vframes[0] = vframes[1]

        # 2) pad/truncate to 150
        base_total = 150
        if T < base_total:
            padding = vframes[-1:].repeat(base_total - T, 1, 1, 1)
            vframes = torch.cat([vframes, padding], dim=0)
        vframes = vframes[:base_total]

        # 3) split 75/75 and pad each side to 77
        context_part = vframes[:75]
        target_part = vframes[75:]

        def _pad_vae(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x[:1], x, x[-1:]], dim=0)

        full_sequence = torch.cat([_pad_vae(context_part), _pad_vae(target_part)], dim=0)
        target_total = full_sequence.shape[0]  # expected 154

        # 4) convert to [0,1]
        vframes = full_sequence
        vframes = vframes.float() / 255.0

        if end_frame is None:
            end_frame = target_total

        return vframes[start_frame:end_frame]

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        video_path = Path(video_metadata["video_paths"])
        candidate_dirs = [video_path.parent, video_path.parent.parent]

        target_json = None
        for parent_dir in candidate_dirs:
            if not parent_dir.exists():
                continue
            for f in parent_dir.iterdir():
                if (
                    f.suffix == ".json"
                    and "_character.json" not in f.name
                    and "_check.json" not in f.name
                ):
                    target_json = f
                    break
            if target_json is not None:
                break

        raw_matrices: List[torch.Tensor] = []
        if target_json:
            try:
                with open(target_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Align with vae_process_dataset.py camera logic (149 -> 150 -> 77/77).
                for i in range(149):
                    key = str(i)
                    if key in data:
                        mat = _parse_camera_matrix(data[key])
                    else:
                        mat = raw_matrices[-1] if raw_matrices else torch.eye(4)
                    raw_matrices.append(mat)
            except Exception as e:
                print(f"Error reading camera json {target_json}: {e}")
                raw_matrices = [torch.eye(4)] * 149
        else:
            raw_matrices = [torch.eye(4)] * 149

        # replace 0 with 1
        if len(raw_matrices) > 1:
            raw_matrices[0] = raw_matrices[1]

        # pad to 150
        raw_matrices.append(raw_matrices[-1])

        # split 75/75 and pad each side (+1 front, +1 back) => 77/77
        context_mats = raw_matrices[:75]
        target_mats = raw_matrices[75:]
        context_final = [context_mats[0]] + context_mats + [context_mats[-1]]
        target_final = [target_mats[0]] + target_mats + [target_mats[-1]]
        matrices = torch.stack(context_final + target_final).view(154, -1)

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
