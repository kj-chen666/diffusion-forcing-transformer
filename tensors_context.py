import copy
import os
# os.environ["CUDA_VIDIBLE_DEVICES"] = "1"
import re
import torch, os, imageio, argparse

from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict
from diffsynth.models.wan_video_dit import SelfAttention, SliceAttentionModule
from diffsynth.models.wan_video_camera_controller import SimpleAdapter
from diffsynth.models.ucpe_attention import UcpeSelfAttention
import torchvision
from PIL import Image
import numpy as np
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import shutil
from context_mem_dataset import ContextMemorySegmentsDataset
from typing import Optional, Dict, Any, List, Tuple
from diffsynth.models.wan_video_dit import SparseFrameAttentionModule

def _parse_caption_line(processed_dir: str, line: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse caption line format: subfolder/video_name.mp4    caption_text
    Returns: (subfolder_name, video_filename, caption)
    """
    line = line.strip()
    if not line:
        return None
    
    # Split on tab or multiple spaces
    parts = re.split(r'\t+', line, maxsplit=1)
    if len(parts) < 2:
         parts = re.split(r'\s{2,}', line, maxsplit=1)
         print("Warning: Line split by multiple spaces")
         print(parts)
    if len(parts) != 2:
        return None
    
    video_path, caption = parts
    
    is_extend = "Extend" in video_path
    
    # Parse subfolder/video_name.mp4
    if not video_path.endswith('.mp4'):
        return None
    
    dir_path, filename = os.path.split(video_path)
    
    contains_hemi8 = "Hemi8" in filename
    # 3. 从后往前获取最后三个层级的路径
    path_parts = []
    count = 0
    
    # 从后往前遍历路径
    while dir_path and count < 3:
        dir_path, folder = os.path.split(dir_path)
        if folder:  # 确保不是空字符串
            path_parts.insert(0, folder)  # 在列表开头插入，保持顺序
            count += 1
    
    # 4. 构建新的路径
    if path_parts:
        new_path = "-".join(path_parts) + "-" + filename
    else:
        new_path = filename
    
    scene = new_path.split("_")[0]
    output_filename = f"{scene}_{filename}.tensors.pth"
    if is_extend:
        output_filename = f"Extend_{scene}_{filename}.tensors.pth"
    output_path = os.path.join(processed_dir, output_filename)


    hemi_idx = None
    hemi_name = None
    if contains_hemi8:
        hemi_idx = "-".join(path_parts) + "-" + filename.split("_C_0")[0]  # 选出8机位之前的运镜way
        if is_extend:
            hemi_idx = f"Extend_{hemi_idx}"

        inter_part = filename.split("_C_")[1]
        inter_part = inter_part.split("mm_")[0]
        hemi_name = "C_" + inter_part + "mm"
    

    return output_path, contains_hemi8, hemi_idx, video_path, hemi_name

def obtain_slice_idx(check_path: str, hemi_idx: str, contains_hemi8: bool) -> int:
    slice_idx = []
    with open(check_path, "r") as f:
        data = json.load(f)
        time_idxs = []
        if contains_hemi8:
            content = data[hemi_idx]
            select_key = list(content.keys())[0]
            content = content[select_key]
        else:
            select_key = list(data.keys())[0]
            content = data[select_key]
            iner_key = list(content.keys())[0]
            content = content[iner_key]
        
        for dic_item in content:
            time_idx = dic_item["frame"]
            if time_idx < 75:
                time_idx += 1
            else:
                time_idx += 3
            
            if time_idx < 60:
                continue
            else:
                time_idxs.append(time_idx)

        if len(time_idxs) == 1:
            slice_idx.append(time_idxs[0])
            if time_idxs[0] < 115:
                slice_idx.append(115)
            else:
                slice_idx.append(78)
        elif len(time_idxs) == 0:
            slice_idx.append(77)
            slice_idx.append(115)
        else:
            slice_idx.append(time_idxs[0])
            slice_idx.append(time_idxs[-1])
            
    return slice_idx



class LightningModelForDataProcess(pl.LightningModule):
    """
    Lightning model for preprocessing Context-as-Memory-Dataset.
    
    Processes:
    1. Video frames -> VAE latents (81 frames)
    2. Context frames -> Individual VAE latents (21 frames) 
    3. Caption -> Text encoder embeddings
    4. First frame -> Image encoder embeddings
    5. Camera data -> Maintains original format
    """
    
    def __init__(
        self, 
        text_encoder_path: str,
        vae_path: str, 
        image_encoder_path: Optional[str] = None,
        tiled: bool = False, 
        tile_size: tuple = (34, 34), 
        tile_stride: tuple = (18, 16),
        output_path: str = "./processed_data"
    ):
        super().__init__()
        
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device='cpu')
        model_manager.load_models(model_path)
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.output_dir = output_path
    
    def encode_video_batch(self, video_tensor: torch.Tensor) -> torch.Tensor:
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized. Please check model loading.")
        
        
        # Move to correct device and dtype
        video_tensor = video_tensor.to(dtype=self.pipe.torch_dtype, device=self.device)
        
        # Encode video to latents
        latents = self.pipe.encode_video(video_tensor, **self.tiler_kwargs)[0]
        
        return latents
    
    def encode_context_frames_individually(self, context_frames: torch.Tensor) -> torch.Tensor:
        # Use batch video encoding for context frames as well (77 frames -> VAE -> 20 latents)
        return self.encode_video_batch(context_frames)
    
    def encode_text_prompt(self, caption: str) -> Dict[str, Any]:
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized. Please check model loading.")
            
        prompt_emb = self.pipe.encode_prompt(caption)
        return prompt_emb
    
    def encode_first_frame_image(self, first_frame: np.ndarray, num_frames: int, height: int, width: int) -> Dict[str, Any]:
        if self.pipe is None:
            return {}
            
        # Convert tensor to PIL Image
        # Assuming first_frame is in [-1, 1] range, convert to [0, 1]
        first_frame_pil = Image.fromarray(first_frame)
        
        # Encode image
        image_emb = self.pipe.encode_image(first_frame_pil, num_frames, height, width)
        
        return image_emb
        # first_frame_normal
    ###############################################################

    def configure_optimizers(self):
        """Lightning钩子：确保Pipeline在正确设备上并同步device属性"""
        if self.pipe is not None:
            # 移动Pipeline对象到GPU
            self.pipe = self.pipe.to(self.device)
            # 关键：同步Pipeline的device属性
            self.pipe.device = self.device
            print(f"Pipeline moved to device: {self.device}")
            print(f"Pipeline.device synchronized: {self.pipe.device}")
        return None 
    
    def on_train_start(self):
        """备份钩子：确保Pipeline设备同步"""
        if self.pipe is not None:
            self.pipe = self.pipe.to(self.device)
            self.pipe.device = self.device
            print(f"Pipeline device backup sync: {self.pipe.device}")
    

    def _ensure_pipeline_device_sync(self):
        """确保Pipeline.device与Lightning.device同步"""
        if self.pipe is not None and self.pipe.device != self.device:
            # print(f"Syncing pipeline device: {self.pipe.device} -> {self.device}")
            self.pipe.device = self.device

    ###############################################################

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:

        # Extract batch data
        self._ensure_pipeline_device_sync()

        captions = batch["caption"]  # List of strings
        videos = batch["video"]      # [B, C, 77, H, W]
        context_frames = batch["context_frames"]  # [B, C, 77, H, W]
        # first_frames = batch["first_frame"]       # [B, C, H, W]
        video_cameras = batch["video_camera"]     # [B, 77, 4, 4]
        context_cameras = batch["context_camera"] # [B, 77, 4, 4]
        meta = batch["meta"]         # Batch metadata
        
        batch_size = videos.shape[0]
        
        # Process each item in the batch
        for i in range(batch_size):
            # Create output filename based on scene and frame range
            scene = meta["scene"][i]
            # For new dataset, we might use filename or other ID
            if "filename" in meta:
                 filename_base = meta["filename"][i]
                 output_filename = f"{scene}_{filename_base}.tensors.pth"
            else:
                 start_frame = meta["extended_start"][i]
                 end_frame = meta["extended_end"][i]
                 output_filename = f"{scene}_{start_frame:04d}_{end_frame:04d}.tensors.pth"
            
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Skip if file already exists
            if os.path.exists(output_path):
                print(f"File {output_path} already exists, skipping.")
                continue
            
            try:
                # Process video frames (Target)
                video_latents = self.encode_video_batch(videos[i:i+1])  # Keep batch dimension
                video_latents = video_latents.squeeze(0)  # Remove batch dimension [C, 20, H, W]
                
                # Process context frames (Context)
                context_latents = self.encode_context_frames_individually(context_frames[i:i+1])
                context_latents = context_latents.squeeze(0) # [C, 20, H, W]
                
                # Process text prompt
                prompt_emb = self.encode_text_prompt(captions[i])
                
                # Process cameras: Subsample every 4th frame to match VAE 4x downsampling
                # 77 frames -> indices 0, 4, ..., 76 (20 frames)
                # video_cameras[i] is [77, 4, 4]
                vid_cam_sub = video_cameras[i][::4] # [20, 4, 4]
                ctx_cam_sub = context_cameras[i][::4] # [20, 4, 4]
                
                # Prepare output data
                output_data = {
                    # Video latents
                    "video_latents": video_latents,           # [C_latent, 20, H_latent, W_latent]
                    "context_latents": context_latents,       # [C_latent, 20, H_latent, W_latent]
                    
                    # Embeddings
                    "prompt_emb": prompt_emb,                 # Text encoder output
                    # "image_emb": image_emb,                   # Image encoder output
                    
                    # Camera data (subsampled tensors)
                    "video_camera": vid_cam_sub,           # [20, 4, 4]
                    "context_camera": ctx_cam_sub,         # [20, 4, 4]
                    
                    "meta": {k: v[i] for k, v in meta.items() if isinstance(v, (list, torch.Tensor))}
                }
                
                # Save to file
                torch.save(output_data, output_path)
                print(f"Saved: {output_path}")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}, item {i}: {e}")
                continue


class TensorDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        processed_dir: str,
        steps_per_epoch: int,
        metadata_csv: Optional[str] = None,
        use_25_percent_data: bool = False,
        train_caption_path: Optional[str] = None,
        improve_selection: bool = False,
        add_ref: bool = False,
        adaptive_slice: bool = False,
    ) -> None:
        super().__init__()
        # Discover processed files
        if not os.path.isfile(train_caption_path):
            raise FileNotFoundError(f"Captions file not found: {train_caption_path}")
        self.add_ref = add_ref
        visited_samples = set()
        self.paths = []
        self.adaptive_slice = adaptive_slice
        if self.adaptive_slice:
            self.slices = []

        with open(train_caption_path, "r", encoding="utf-8") as f:
            for line in f:
                sample_path, contains_hemi8, hemi_idx, video_path, hemi_name = _parse_caption_line(processed_dir, line)
                if not os.path.exists(sample_path):
                    print(f"Warning: {sample_path} does not exist.")
                    continue

                if self.adaptive_slice:
                    if contains_hemi8:
                        check_path = video_path.replace("videos/", "")
                        check_path = check_path.split(hemi_name)[0] + "check.json"
                    else:
                        check_path = video_path.split("/videos/")[0]
                        check_root, check_name = os.path.split(check_path)
                        check_name = check_name + "_check.json"
                        check_path = os.path.join(check_path, check_name)

                    if not os.path.exists(check_path):
                        print(f"Warning: {check_path} does not exist.")
                        continue
                    slice_idx = obtain_slice_idx(check_path, hemi_name, contains_hemi8)
                    self.slices.append(slice_idx)
                
                if sample_path:
                    if not improve_selection:
                        self.paths.append(sample_path)
                    else:    # eliminate redundant hemi8 samples to improve training efficiency
                        if hemi_idx != None:
                            if hemi_idx not in visited_samples:
                                self.paths.append(sample_path)
                                visited_samples.add(hemi_idx)
                            else:
                                print(f"Already visited {hemi_idx}, skipping {sample_path}.")
                        else:
                            self.paths.append(sample_path)
                

        # self.paths = [
        #     os.path.join(processed_dir, f)
        #     for f in os.listdir(processed_dir)
        #     if f.endswith(".tensors.pth")
        # ]
        ###### TODO 取百分之25数据
        # self.paths = self.paths[:int(len(self.paths) * 0.25)]
        if use_25_percent_data:
            print(f"Using 25% of data: {len(self.paths)} -> {int(len(self.paths) * 0.25)}")
            self.paths = self.paths[:int(len(self.paths) * 0.25)]

        if len(self.paths) == 0:
            raise RuntimeError(f"No processed .tensors.pth files found in {processed_dir}")

        print(len(self.paths), "processed tensors found.")
        self.steps_per_epoch = steps_per_epoch
        # self.cam_idx is not needed if cameras are already subsampled
        self.cam_idx: List[int] = list(range(77))[::4]

    # ------------------------------
    # Camera utilities
    # ------------------------------
    @staticmethod
    def _deg2rad(deg: float) -> float:
        return deg * 3.141592653589793 / 180.0

    @classmethod
    def _euler_xyz_to_rot(cls, euler_deg: List[float]) -> torch.Tensor:
        """
        Build rotation matrix from Euler angles (rotation order X->Y->Z) in degrees.
        euler = [x, y, z] where these are rotations around X, Y, Z axes respectively.
        Returns [3,3] torch tensor (float32).
        """
        x_deg, y_deg, z_deg = euler_deg
        x, y, z = cls._deg2rad(x_deg), cls._deg2rad(y_deg), cls._deg2rad(z_deg)

        cx, sx = torch.cos(x), torch.sin(x)
        cy, sy = torch.cos(y), torch.sin(y)
        cz, sz = torch.cos(z), torch.sin(z)

        # R = Rz @ Ry @ Rx (common convention); adjust if your data uses different order
        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
        R = Rz @ Ry @ Rx
        return R

    @classmethod
    def _pose_to_c2w(cls, position: List[float], rotation: List[float]) -> torch.Tensor:
        """
        Build 4x4 C2W from position [x,y,z] and rotation Euler [x,y,z] degrees.
        """
        R = cls._euler_xyz_to_rot(rotation)  # [3,3]
        t = torch.tensor(position, dtype=torch.float32).view(3, 1)  # [3,1]
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = R
        c2w[:3, 3:] = t
        return c2w

    @staticmethod
    def _apply_coordinate_transform(c2w: torch.Tensor) -> torch.Tensor:
        """
        Apply axis reordering and scaling to match reference pipeline:
          - reorder columns: [Y, Z, X, W] for 4x4 matrix columns
          - flip Y axis (second column) sign for first 3 rows
          - scale translation by 1/100
        Returns transformed 4x4 tensor (float32).
        """
        # Reorder columns [1,2,0,3]
        c2w_t = c2w.clone()
        c2w_t = c2w_t[:, [1, 2, 0, 3]]
        # Flip Y axis (column index 1 in rotation part)
        c2w_t[:3, 1] *= -1.0
        # Scale translation
        c2w_t[:3, 3] /= 100.0
        return c2w_t

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            try:
                # Fixed-seed like cycling
                data_id = torch.randint(0, len(self.paths), (1,))[0]
                data_id = int((data_id + index) % len(self.paths))
                path_tgt = self.paths[data_id]

                slice_idx = {}
                if self.adaptive_slice:
                    slice_idx = self.slices[data_id]

                sample = torch.load(path_tgt, map_location="cpu")
                if self.add_ref:
                    ref_path = path_tgt.replace("full_", "ref_")
                    ref_sample = torch.load(ref_path, map_location="cpu")
                    ref_latents: torch.Tensor = ref_sample["ref_latents"]  # [C, 1, H, W]

                video_latents: torch.Tensor = sample["video_latents"]  # [C, 81, H, W]
                context_latents: torch.Tensor = sample["context_latents"]  # [C, 21, H, W]
                prompt_emb: Dict[str, torch.Tensor] = sample.get("prompt_emb", {})
                image_emb: Dict[str, torch.Tensor] = sample.get("image_emb", {})
                video_camera_obj = sample.get("video_camera")  # list of 81 dicts OR [81,12] tensor
                context_camera_obj = sample.get("context_camera")  # list of 21 dicts OR [21,12] tensor
        
                # 1) Ref C2W from first frame of video (index 0)
                # Handle tensor input [20, 4, 4]
                if torch.is_tensor(video_camera_obj) and video_camera_obj.ndim == 3 and video_camera_obj.shape[-2:] == (4, 4):
                    # Tensor case: [20, 4, 4]
                    ref_c2w = video_camera_obj[0].clone()
                    ref_c2w = self._apply_coordinate_transform(ref_c2w)
                    ref_w2c = torch.inverse(ref_c2w)
                elif isinstance(video_camera_obj, list) and len(video_camera_obj) >= 1 and isinstance(video_camera_obj[0], dict):
                    # Legacy Dict case
                    ref_c2w = self._pose_to_c2w(
                        video_camera_obj[0].get("position", [0.0, 0.0, 0.0]),
                        video_camera_obj[0].get("rotation", [0.0, 0.0, 0.0])
                    )
                    ref_c2w = self._apply_coordinate_transform(ref_c2w)
                    ref_w2c = torch.inverse(ref_c2w)
                else:
                    ref_w2c = None

                # 2) Video relative
                if torch.is_tensor(video_camera_obj) and video_camera_obj.ndim == 3 and video_camera_obj.shape[-2:] == (4, 4) and ref_w2c is not None:
                    # [20, 4, 4] -> Compute relative poses
                    rel_list_v = []
                    for i in range(len(video_camera_obj)):
                         c2w = video_camera_obj[i].clone()
                         c2w = self._apply_coordinate_transform(c2w)
                         rel = ref_w2c @ c2w
                         rel_list_v.append(rel[:3, :4].contiguous().view(-1))
                    cam_video_rel = torch.stack(rel_list_v, dim=0) # [20, 12]
                elif isinstance(video_camera_obj, list) and len(video_camera_obj) >= 81 and isinstance(video_camera_obj[0], dict) and ref_w2c is not None:
                    # Legacy list subsampling
                    cams_sub = [video_camera_obj[i] for i in range(len(video_camera_obj))]
                    rel_list_v = []
                    for cam in cams_sub:
                        c2w = self._pose_to_c2w(
                            cam.get("position", [0.0, 0.0, 0.0]),
                            cam.get("rotation", [0.0, 0.0, 0.0])
                        )
                        c2w = self._apply_coordinate_transform(c2w)
                        rel = ref_w2c @ c2w 
                        rel_list_v.append(rel[:3, :4].contiguous().view(-1))
                    cam_video_rel = torch.stack(rel_list_v, dim=0)
                elif torch.is_tensor(video_camera_obj) and video_camera_obj.shape[0] in (81, 21, 20):
                     # Pre-computed tensor case
                     cam_video_rel = video_camera_obj
                else:
                    print("Warning: video_camera missing or invalid, using zeros.")
                    cam_video_rel = torch.zeros(20, 12, dtype=torch.float32)

                # 3) Context relative
                if torch.is_tensor(context_camera_obj) and context_camera_obj.ndim == 3 and context_camera_obj.shape[-2:] == (4, 4) and ref_w2c is not None:
                    # [20, 4, 4] -> Compute relative poses
                    rel_list_c = []
                    for i in range(len(context_camera_obj)):
                         c2w = context_camera_obj[i].clone()
                         c2w = self._apply_coordinate_transform(c2w)
                         rel = ref_w2c @ c2w
                         rel_list_c.append(rel[:3, :4].contiguous().view(-1))
                    cam_context_rel = torch.stack(rel_list_c, dim=0) # [20, 12]
                elif isinstance(context_camera_obj, list) and len(context_camera_obj) >= 1 and isinstance(context_camera_obj[0], dict) and ref_w2c is not None:
                    rel_list_c = []
                    # Expect 21 context frames; if not, take first 21 or pad
                    ctx_list = [context_camera_obj[i] for i in range(len(context_camera_obj))]
                    for cam in ctx_list:
                        c2w = self._pose_to_c2w(
                            cam.get("position", [0.0, 0.0, 0.0]),
                            cam.get("rotation", [0.0, 0.0, 0.0])
                        )
                        c2w = self._apply_coordinate_transform(c2w)
                        rel = ref_w2c @ c2w
                        rel_list_c.append(rel[:3, :4].contiguous().view(-1))
                    # Pad to 21 if shorter
                    while len(rel_list_c) < 21:
                        rel_list_c.append(torch.zeros(12, dtype=torch.float32))
                    cam_context_rel = torch.stack(rel_list_c[:21], dim=0)
                elif torch.is_tensor(context_camera_obj) and context_camera_obj.shape[0] in (21, 20):
                    cam_context_rel = context_camera_obj
                else:
                    print("Warning: context_camera missing or invalid, using zeros.")
                    cam_context_rel = torch.zeros(20, 12, dtype=torch.float32)

                # Concatenate along time dimension to [C, 42, H, W]
                latents = torch.cat((context_latents, video_latents), dim=1)
                if self.add_ref:
                    latents = torch.cat((ref_latents, latents), dim=1)

                data: Dict[str, Any] = {
                    "latents": latents,  # [C, 42, H, W]
                    # Relative cameras w.r.t target first frame (both streams)
                    "video_camera": cam_video_rel.to(torch.bfloat16),      # [21,12]
                    "context_camera": cam_context_rel.to(torch.bfloat16),  # [21,12]
                    "prompt_emb": prompt_emb,
                    "image_emb": image_emb if isinstance(image_emb, dict) else {},
                    "meta": sample.get("meta", {}),
                    "slice_idx": slice_idx,
                }

                return data
            except Exception as e:
                # Fallback to another index on transient read errors
                print(f"Error loading or processing data: {e}, trying another index.")
                index = random.randrange(len(self.paths))




class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
        add_ref=False,
        use_PRoPE=False,
        use_UCPE=False,
        adaptive_slice=False,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.use_PRoPE = use_PRoPE
        self.use_UCPE = use_UCPE
        self.pipe.dit.use_PRoPE = use_PRoPE
        
        if use_PRoPE:
            patch_size = self.pipe.dit.patch_size
            self.pipe.dit.camera_adapter = SimpleAdapter(24, 1536, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.pipe.dit.camera_adapter = None

        dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        num_heads=self.pipe.dit.blocks[0].self_attn.num_heads
        head_dim=dim//num_heads
        self.pipe.use_UCPE = use_UCPE
        self.pipe.dit.use_UCPE = use_UCPE
        self.pipe.dit.modify_rope = args.modify_rope
        self.add_ref = add_ref
        self.adaptive_slice = adaptive_slice
        self.pipe.adaptive_slice = adaptive_slice
        self.pipe.dit.adaptive_slice = adaptive_slice

        for block in self.pipe.dit.blocks:
            block.use_UCPE = use_UCPE
            block.self_attn.use_UCPE = use_UCPE

            block.adaptive_slice = adaptive_slice
            block.self_attn.adaptive_slice = adaptive_slice


            block.cam_encoder_con = nn.Linear(12, dim)
            block.cam_encoder_tgt = nn.Linear(12, dim)
            block.projector = nn.Linear(dim, dim)
            # block.sparse_projector = nn.Linear(dim, dim)
            block.cam_encoder_con.weight.data.zero_()
            block.cam_encoder_con.bias.data.zero_()
            block.cam_encoder_tgt.weight.data.zero_()
            block.cam_encoder_tgt.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))

            if use_PRoPE:
                block.self_attn.rope_phase_qk = nn.Sequential(nn.SiLU(), nn.Linear(dim, block.num_heads * (2 * head_dim // 3 // 2), bias=False))
                block.self_attn.rope_phase_vo = nn.Sequential(nn.SiLU(), nn.Linear(dim, block.num_heads * (2 * head_dim // 3 // 2), bias=False))
                block.self_attn.rope_phase_qk[-1].weight.data.zero_()
                block.self_attn.rope_phase_vo[-1].weight.data.zero_()
            # block.cam_encoder_con = nn.Linear(12, dim).to(dtype=self.pipe.torch_dtype)
            # block.cam_encoder_tgt = nn.Linear(12, dim).to(dtype=self.pipe.torch_dtype)
            # block.projector = nn.Linear(dim, dim).to(dtype=self.pipe.torch_dtype)
            # # block.sparse_projector = nn.Linear(dim, dim)
            # block.cam_encoder_con.weight.data.zero_()
            # block.cam_encoder_con.bias.data.zero_()
            # block.cam_encoder_tgt.weight.data.zero_()
            # block.cam_encoder_tgt.bias.data.zero_()
            # block.projector.weight = nn.Parameter(torch.eye(dim).to(dtype=self.pipe.torch_dtype))
            # block.projector.bias = nn.Parameter(torch.zeros(dim).to(dtype=self.pipe.torch_dtype))
            # block.sparse_projector.weight = nn.Parameter(torch.eye(dim))
            # block.sparse_projector.bias = nn.Parameter(torch.zeros(dim))

            if use_UCPE:
                width = 832
                height = 480
                attn_compress=1
                patch_factor = 16 # self.pipe.vae.upsampling_factor * 2
                patches_x = width // patch_factor
                patches_y = height // patch_factor
                emb_dim = None
                block.cam_self_attn = UcpeSelfAttention(
                    self.pipe.dit.dim,
                    self.pipe.dit.dim // attn_compress,
                    block.num_heads // attn_compress,
                    patches_x=patches_x,
                    patches_y=patches_y,
                    image_width=width,
                    image_height=height,
                    emb_dim=emb_dim,
                    adaptation_method="parallel",
                )
                block.use_UCPE = use_UCPE
                block.self_attn.use_UCPE = use_UCPE

            block.modify_rope = args.modify_rope
            block.self_attn.change_sparse = args.change_sparse
            block.change_sparse = args.change_sparse
            block.add_ref = add_ref
            if args.change_sparse:
                block.self_attn.attention_type = "sparse_frame"
                block.self_attn.attn = SparseFrameAttentionModule(
                num_heads=num_heads,
                num_frames=40,
                frame_hw= (30 * 52),
                top_k=9,
                frame_chunk_size=None  # 显存优化：分批处理帧
            )

            if self.adaptive_slice:
                block.slice_attention = SelfAttention(
                dim, num_heads, 1e-6, "slice",
                # sparse_frame_args={"use_sink_token": args.use_sink_token}
                )
                block.slice_attention.use_UCPE = use_UCPE
                self._copy_attention_weights(block.slice_attention, block.self_attn)
            # block.use_sparse_attn = args.use_sparse_attn
            # block.sparse_attn = SelfAttention(
            #     dim, num_heads, 1e-6, "sparse_frame",
            #     sparse_frame_args={"use_sink_token": args.use_sink_token}
            # ) # 1536, 12, 1e-6
            # block.block_split = args.block_split
            # block.sparse_attn = SelfAttention(dim, num_heads, 1e-6,"sparse_frame") # 1536, 12, 1e-6
            # self._copy_attention_weights(block.sparse_attn, block.self_attn)

        
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)
            print("Checkpoint loaded successfully")

        self.freeze_parameters()
        if not use_UCPE:
            for name, module in self.pipe.denoising_model().named_modules():
                if any(keyword in name for keyword in ["cam_encoder_con", "cam_encoder_tgt", "projector", "self_attn", "sparse_attn", "sparse_projector", "camera_adapter", "slice_attention"]):
                    print(f"Trainable: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            for name, module in self.pipe.denoising_model().named_modules():
                if any(keyword in name for keyword in ["cam_self_attn", "cam_encoder_con", "cam_encoder_tgt"]):
                    print(f"Trainable: {name}")
                    for param in module.parameters():
                        param.requires_grad = True

        trainable_params = 0
        seen_params = set()
        for name, module in self.pipe.denoising_model().named_modules():
            for param in module.parameters():
                if param.requires_grad and param not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(param)
        print(f"Total number of trainable parameters: {trainable_params}")
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
    def _copy_attention_weights(self, new_attn, pretrained_attn):
        """复制预训练的attention权重到新的attention层"""
        # 复制Q、K、V的权重和偏置
        new_attn.q.weight.data.copy_(pretrained_attn.q.weight.data)
        if hasattr(pretrained_attn.q, 'bias') and pretrained_attn.q.bias is not None:
            new_attn.q.bias.data.copy_(pretrained_attn.q.bias.data)
        
        new_attn.k.weight.data.copy_(pretrained_attn.k.weight.data)
        if hasattr(pretrained_attn.k, 'bias') and pretrained_attn.k.bias is not None:
            new_attn.k.bias.data.copy_(pretrained_attn.k.bias.data)
        
        new_attn.v.weight.data.copy_(pretrained_attn.v.weight.data)
        if hasattr(pretrained_attn.v, 'bias') and pretrained_attn.v.bias is not None:
            new_attn.v.bias.data.copy_(pretrained_attn.v.bias.data)
        
        # 复制输出投影层的权重和偏置（注意：这里应该是 'o' 而不是 'proj'）
        new_attn.o.weight.data.copy_(pretrained_attn.o.weight.data)
        if hasattr(pretrained_attn.o, 'bias') and pretrained_attn.o.bias is not None:
            new_attn.o.bias.data.copy_(pretrained_attn.o.bias.data)
        
        # 如果还有norm层，也需要复制（根据你的初始化代码）
        if hasattr(new_attn, 'norm_q') and hasattr(pretrained_attn, 'norm_q'):
            new_attn.norm_q.weight.data.copy_(pretrained_attn.norm_q.weight.data)
            if hasattr(pretrained_attn.norm_q, 'bias') and pretrained_attn.norm_q.bias is not None:
                new_attn.norm_q.bias.data.copy_(pretrained_attn.norm_q.bias.data)
        
        if hasattr(new_attn, 'norm_k') and hasattr(pretrained_attn, 'norm_k'):
            new_attn.norm_k.weight.data.copy_(pretrained_attn.norm_k.weight.data)
            if hasattr(pretrained_attn.norm_k, 'bias') and pretrained_attn.norm_k.bias is not None:
                new_attn.norm_k.bias.data.copy_(pretrained_attn.norm_k.bias.data)


    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
    
    def get_plucker_emb(self, batch):
        camera_input = torch.cat([batch["context_camera"], batch["video_camera"]], dim=1)
        cond_plucker_embedding = self.pipe.dit.camera_adapter.process_camera_coordinates(
            None, 77, 480, 832, None, None, camera_input# batch["video_camera"]
        )
        cond_plucker_embedding = cond_plucker_embedding[:, :77].permute([0, 4, 1, 2, 3])
        cond_plucker_embedding = torch.repeat_interleave(cond_plucker_embedding, repeats=4, dim=2)
        # cond_plucker_embedding = torch.concat(
        #     [
        #         torch.repeat_interleave(cond_plucker_embedding[:, :, 0:1], repeats=4, dim=2),
        #         cond_plucker_embedding[:, :, 1:]
        #     ], dim=2
        # )
        cond_plucker_embedding = rearrange(cond_plucker_embedding, 'b c (f k) h w -> b (c k) f h w', k=4)
        cond_plucker_embedding = cond_plucker_embedding.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
        return cond_plucker_embedding

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device, dtype=self.pipe.torch_dtype)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][:, 0].to(self.device, dtype=self.pipe.torch_dtype) ### TODO ????????为何只选了第一个
        image_emb = {}

        frame_ids = batch["slice_idx"]
        if self.adaptive_slice:
            frame_ids = None
        
        cond_plucker_embedding = None
        if self.use_PRoPE:
            cond_plucker_embedding = self.get_plucker_emb(batch)
        
        if self.use_UCPE:
            cond_plucker_embedding = torch.cat([batch["context_camera"], batch["video_camera"]], dim=1)
        # inputs_shared['cond_plucker_embedding'] = cond_plucker_embedding
        # if "clip_feature" in image_emb:
        #     image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        # if "y" in image_emb:
        #     image_emb["y"] = image_emb["y"][0].to(self.device)
        
        cam_emb_tgt = batch["video_camera"].to(self.device, dtype=self.pipe.torch_dtype)
        cam_emb_con = batch["context_camera"].to(self.device, dtype=self.pipe.torch_dtype)

        # Loss
        self.pipe.device = self.device
        extra_input = {}# self.pipe.prepare_extra_input(latents) # No extra input
        # origin_latents = copy.deepcopy(latents)
        # with torch.no_grad():
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        tgt_latent_len = noisy_latents.shape[2] // 2
        if self.add_ref:
            tgt_latent_len += 1

        noisy_latents[:, :, :tgt_latent_len, ...] = latents[:, :, :tgt_latent_len, ...]
        training_target = self.pipe.scheduler.training_target(latents[:, :, tgt_latent_len:, ...], noise[:, :, tgt_latent_len:, ...], timestep)
        weight = self.pipe.scheduler.training_weight(timestep)
        
        del noise
        
        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, cam_emb_tgt=cam_emb_tgt, cam_emb_con=cam_emb_con, **prompt_emb, cond_plucker_embedding=cond_plucker_embedding, frame_ids=frame_ids, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        
        
        
        loss = torch.nn.functional.mse_loss(noise_pred[:, :, tgt_latent_len:, ...].float(), training_target.float())
        loss = loss * weight

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        # if checkpoint_dir is None:
            # checkpoint_dir = os.path.join(self.trainer.default_root_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))



def parse_args():
    parser = argparse.ArgumentParser(description="Train ReCamMaster")
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        # required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/m2v_intern_v3/chenkaijin/full_processed",
        # required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./dbug",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="/m2v_intern/chenkaijin/ckpt/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="/m2v_intern/chenkaijin/ckpt/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="/m2v_intern/chenkaijin/ckpt/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=True,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=32000,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=77,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10000,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=True,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.csv",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes for distributed training.",
    )
    parser.add_argument(
        "--use_25_percent_data",
        action="store_true",
        default=False,
        help="Use only 25% of the data.",
    )
    parser.add_argument(
        "--use_sparse_attn",
        action="store_true",
        default=False,
        help="Whether to use sparse attention.",
    )
    parser.add_argument(
        "--use_sink_token",
        action="store_true",
        default=False,
        help="Whether to use sink token in sparse attention.",
    )
    parser.add_argument(
        "--block_split",
        action="store_true",
        default=False,
        help="Whether to split self attention and sparse attention",
    )
    parser.add_argument(
        "--modify_rope",
        action="store_true",
        default=False,
        help="Whether to modify rope",
    )
    parser.add_argument(
        "--change_sparse",
        action="store_true",
        default=True,
        help="Whether to modify rope",
    )
    parser.add_argument(
        "--improve_selection",
        action="store_true",
        default=False,
        help="improve selection strategy during training",
    )
    parser.add_argument(
        "--train_caption_path",
        type=str,
        default="/m2v_intern_v3/chenkaijin/train_shuffled.txt",
        help="Captions of train set",
    )
    parser.add_argument(
        "--add_ref",
        action="store_true",
        default=False,
        help="add reference latent",
    )
    parser.add_argument(
        "--PRoPE",
        action="store_true",
        default=True,
        help="use PRoPE",
    )
    parser.add_argument(
        "--UCPE",
        action="store_true",
        default=False,
        help="use UCPE",
    )
    parser.add_argument(
        "--adaptive_slice",
        action="store_true",
        default=False,
        help="use adaptive slice",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = ContextMemorySegmentsDataset(
        root_dir=args.dataset_path,
        height=args.height,
        width=args.width,
        seed=42
    )
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # Process in order
        num_workers=args.dataloader_num_workers,
        pin_memory=False,
        drop_last=False

    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        output_path=args.output_path
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch,
        use_25_percent_data=args.use_25_percent_data,
        train_caption_path=args.train_caption_path,
        improve_selection=args.improve_selection,
        add_ref=args.add_ref,
        adaptive_slice=args.adaptive_slice,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
        add_ref=args.add_ref,
        use_PRoPE=args.PRoPE,
        use_UCPE=args.UCPE,
        adaptive_slice=args.adaptive_slice,
    )

    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        num_nodes=args.num_nodes,   # multi_nodes
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)