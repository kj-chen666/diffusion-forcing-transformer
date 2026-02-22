import os
import json
import random
import re
import glob
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from clip_eval import simple_process
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import torchvision
from torchvision.io import read_video
import torch.nn.functional as F

def crop_and_center_zoom(img_tensor, box, visualize=False):
    """
    参数:
    - img_tensor: torch.Tensor, 形状为 (C, 1, H, W)
    - box: tuple/list, 格式为 (x1, y1, x2, y2)，在原图坐标系中
    - visualize: bool, 是否可视化结果
    
    返回:
    - result: torch.Tensor, 处理后的图像
    """
    # 获取输入参数
    C, _, H, W = img_tensor.shape
    
    # 确保box是tensor
    if isinstance(box, (list, tuple)):
        box = torch.tensor(box, dtype=torch.float32)
    
    x1, y1, x2, y2 = box
    
    # 1. 提取box区域
    # 注意：box坐标是 (H, W) 顺序
    box_h = int(y2 - y1)
    box_w = int(x2 - x1)
    if box_h > 0 and box_w > 0:
        # 裁剪box区域
        cropped = img_tensor[:, :, int(y1):int(y2), int(x1):int(x2)]
        
        # 2. 计算放大到图片中心的最大缩放比例
        # 确保不超出原图范围
        if box_h == 0:
            box_h = 1
        if box_w == 0:
            box_w = 1
        scale_h = H / box_h
        scale_w = W / box_w
        scale = min(scale_h, scale_w)
        
        # 3. 计算缩放后的尺寸
        new_h = int(box_h * scale)
        new_w = int(box_w * scale)
        
        # 4. 放大裁剪区域
        # 使用双线性插值进行缩放
        resized = F.interpolate(
            cropped, 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 5. 创建黑色背景
    black_background = torch.zeros_like(img_tensor)
    
    # 6. 计算居中位置
    if box_h > 0 and box_w > 0:
        y_center = (H - new_h) // 2
        x_center = (W - new_w) // 2
    
    # 7. 将放大的区域放到黑色背景的中心
    result = black_background.clone()
    if box_h > 0 and box_w > 0: 
        result[:, :, y_center:y_center+new_h, x_center:x_center+new_w] = resized
    
    # 8. 可视化
    if visualize:
        sp = visualize_result(img_tensor, box, result)
    
    return result

def visualize_result(original_img, box, result_img, save_dir='./results'):
    """
    可视化原始图像、box区域和处理结果
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保存原始图像（带框）
    orig_np = original_img.squeeze(1).cpu().numpy()
    
    if orig_np.shape[0] == 3:  # RGB
        orig_np = np.transpose(orig_np, (1, 2, 0))
        orig_np = (orig_np * 255).astype(np.uint8)  # 假设是0-1范围
    else:  # 灰度图
        orig_np = orig_np[0]
        if orig_np.max() <= 1.0:  # 如果是0-1范围
            orig_np = (orig_np * 255).astype(np.uint8)
        else:
            orig_np = orig_np.astype(np.uint8)
    
    # 绘制红色框
    x1, y1, x2, y2 = [int(coord) for coord in box]
    if len(orig_np.shape) == 3:  # RGB
        # 在RGB图像上画红色框
        orig_with_box = orig_np.copy()
        # 上边框
        orig_with_box[y1:y1+2, x1:x2, 0] = 255
        orig_with_box[y1:y1+2, x1:x2, 1] = 0
        orig_with_box[y1:y1+2, x1:x2, 2] = 0
        # 下边框
        orig_with_box[y2-2:y2, x1:x2, 0] = 255
        orig_with_box[y2-2:y2, x1:x2, 1] = 0
        orig_with_box[y2-2:y2, x1:x2, 2] = 0
        # 左边框
        orig_with_box[y1:y2, x1:x1+2, 0] = 255
        orig_with_box[y1:y2, x1:x1+2, 1] = 0
        orig_with_box[y1:y2, x1:x1+2, 2] = 0
        # 右边框
        orig_with_box[y1:y2, x2-2:x2, 0] = 255
        orig_with_box[y1:y2, x2-2:x2, 1] = 0
        orig_with_box[y1:y2, x2-2:x2, 2] = 0
    else:  # 灰度图
        # 转换为RGB以便画红色框
        orig_with_box = np.stack([orig_np, orig_np, orig_np], axis=-1)
        # 在灰度图像上画红色框
        orig_with_box[y1:y1+2, x1:x2, 0] = 255
        orig_with_box[y1:y1+2, x1:x2, 1] = 0
        orig_with_box[y1:y1+2, x1:x2, 2] = 0
        orig_with_box[y2-2:y2, x1:x2, 0] = 255
        orig_with_box[y2-2:y2, x1:x2, 1] = 0
        orig_with_box[y2-2:y2, x1:x2, 2] = 0
        orig_with_box[y1:y2, x1:x1+2, 0] = 255
        orig_with_box[y1:y2, x1:x1+2, 1] = 0
        orig_with_box[y1:y2, x1:x1+2, 2] = 0
        orig_with_box[y1:y2, x2-2:x2, 0] = 255
        orig_with_box[y1:y2, x2-2:x2, 1] = 0
        orig_with_box[y1:y2, x2-2:x2, 2] = 0
    
    # 保存原始图像
    orig_img = Image.fromarray(orig_with_box)
    orig_img.save(os.path.join(save_dir, 'original_with_box.png'))
    print(f"保存原始图像到: {os.path.join(save_dir, 'original_with_box.png')}")
    
    # 2. 保存裁剪并放大的结果图像
    result_np = result_img.squeeze(1).cpu().numpy()
    
    if result_np.shape[0] == 3:  # RGB
        result_np = np.transpose(result_np, (1, 2, 0))
        if result_np.max() <= 1.0:  # 如果是0-1范围
            result_np = (result_np * 255).astype(np.uint8)
        else:
            result_np = result_np.astype(np.uint8)
    else:  # 灰度图
        result_np = result_np[0]
        if result_np.max() <= 1.0:  # 如果是0-1范围
            result_np = (result_np * 255).astype(np.uint8)
        else:
            result_np = result_np.astype(np.uint8)
    
    result_img_pil = Image.fromarray(result_np)
    result_img_pil.save(os.path.join(save_dir, 'cropped_and_zoomed.png'))
    print(f"保存结果图像到: {os.path.join(save_dir, 'cropped_and_zoomed.png')}")
    
    # 3. 保存裁剪区域（不包含黑色背景）
    if result_np.ndim == 2:  # 灰度图
        non_zero_mask = result_np > 0
    else:  # RGB
        non_zero_mask = result_np.any(axis=2)
    
    rows = np.any(non_zero_mask, axis=1)
    cols = np.any(non_zero_mask, axis=0)
    
    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        cropped_region = result_np[y_min:y_max+1, x_min:x_max+1]
        cropped_img = Image.fromarray(cropped_region)
        cropped_img.save(os.path.join(save_dir, 'cropped_region_only.png'))
        print(f"保存裁剪区域到: {os.path.join(save_dir, 'cropped_region_only.png')}")
    
    return {
        'original_path': os.path.join(save_dir, 'original_with_box.png'),
        'result_path': os.path.join(save_dir, 'cropped_and_zoomed.png'),
        'cropped_path': os.path.join(save_dir, 'cropped_region_only.png')
    }


@dataclass
class CameraData:
    position: List[float]  # [x, y, z]
    rotation: List[float]  # [x, y, z] (yaw, pitch, roll)
    scale: List[float]     # [x, y, z]


def default_frame_transform(height: int, width: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _read_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _parse_caption_line(line: str) -> Optional[Tuple[str, str, str]]:
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
    
    # Parse subfolder/video_name.mp4
    if not video_path.endswith('.mp4'):
        return None
    
    dir_path, filename = os.path.split(video_path)
    
    # 2. 检查文件名是否包含"Hemi8"
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
    
    # Extract subfolder and filename
    # Assuming format: subfolder/filename.mp4
    # if '/' in video_path:
    #     subfolder, filename = video_path.split('/', 1)
    #     # Handle nested paths if any, though user example suggests single level
    #     # If path is A/B/C.mp4, subfolder is A/B
    #     subfolder = os.path.dirname(video_path)
    #     filename = os.path.basename(video_path)
    # else:
    #     return None
    
    return video_path, caption, new_path, filename, contains_hemi8


def _parse_camera_matrix(json_str: str) -> torch.Tensor:
    """
    Parse camera matrix string: "[-1 ... ] [ ... ] [ ... ] [ ... ] "
    Returns 4x4 tensor (transposed from column vectors).
    """
    # Find all occurrences of content within brackets [ ... ]
    matches = re.findall(r'\[(.*?)\]', json_str)
    if len(matches) != 4:
        # Fallback or error
        return torch.eye(4)
    
    cols = []
    for m in matches:
        # Parse space-separated floats
        vals = [float(x) for x in m.strip().split()]
        if len(vals) != 4:
            return torch.eye(4)
        cols.append(vals)
        
    mat = torch.tensor(cols, dtype=torch.float32).T
    return mat
    

def _parse_camera_matrix_hemi8(json_str: str) -> torch.Tensor:
    """
    Parse camera matrix string: "[-1 ... ] [ ... ] [ ... ] [ ... ] "
    Returns 4x4 tensor (transposed from column vectors).
    """
    # Find all occurrences of content within brackets [ ... ]
    matches = re.findall(r'\[(.*?)\]', json_str)
    if len(matches) != 4:
        # Fallback or error
        return torch.eye(4)
    
    cols = []
    for m in matches:
        # Parse space-separated floats
        vals = [float(x) for x in m.strip().split()]
        if len(vals) != 4:
            return torch.eye(4)
        cols.append(vals)
    
    mat = torch.tensor(cols, dtype=torch.float32).T
    return mat


class ContextMemorySegmentsDataset(Dataset):
    """
    Dataset for Context-as-Memory-Dataset with MP4 videos.
    
    Structure:
    - root_dir/
        - captions.txt
        - sample_subfolder/
            - videos/
                - video.mp4 (target)
                - videoDepth.mp4
            - json/
                - ...json (camera params)
    
    Process:
    1. Read video (149 frames) -> 0 replaced by 1 -> Pad last to 150.
    2. Split: Context (0-75), Target (75-150).
    3. Pad VAE: Context (+1 front, +1 back) -> 77, Target (+1 front, +1 back) -> 77.
    4. Cameras: Parse JSON, sync with video ops.
    """
    
    def __init__(
        self,
        root_dir: str,
        captions_path: Optional[str] = None,
        height: int = 480,
        width: int = 832, # Updated default to match user request context if needed, kept 480p
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        
        # Use provided captions path or default
        if captions_path is None:
            captions_path = os.path.join(root_dir, "captions.txt")
        self.captions_path = captions_path
        
        self.height = height
        self.width = width
        # Note: transform is applied to the video tensor
        self.transform = transform
        
        self.samples: List[Tuple[str, str, str, str, bool]] = []  # (video_path, caption, new_path, filename, contains_hemi8)
        
        self.resume = True
        self.output_dir = "/m2v_intern_v3/chenkaijin/full_processed"
        if self.resume:
            self._load_captions_skip_existing()
        else:
            self._load_captions()

        self.rng = random.Random(seed)
        print(f"Loaded {len(self.samples)} samples from captions.txt")
    
    def _load_captions(self):
        """Load and parse captions.txt file."""
        if not os.path.isfile(self.captions_path):
            raise FileNotFoundError(f"Captions file not found: {self.captions_path}")
        
        with open(self.captions_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_caption_line(line)
                if parsed:
                    self.samples.append(parsed)
    
    def _load_captions_skip_existing(self):
        """Load and parse captions.txt file."""
        if not os.path.isfile(self.captions_path):
            raise FileNotFoundError(f"Captions file not found: {self.captions_path}")
        
        with open(self.captions_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_caption_line(line)
                video_path, caption, new_name, filename, is_hemi8 = parsed
                
                scene = new_name.split("_")[0]
                if "Extend" in new_name:
                    output_filename = f"Extend_{scene}_{filename}.tensors.pth"
                else:    
                    output_filename = f"{scene}_{filename}.tensors.pth"
                output_path = os.path.join(self.output_dir, output_filename)
                if os.path.exists(output_path):
                    print(f"File {output_path} already exists, skipping.")
                    continue

                if parsed:
                    self.samples.append(parsed)     

    def _load_video_frames(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video, process frames, split into context and target.
        Returns: (context_frames [C, 77, H, W], target_frames [C, 77, H, W])
        """
        # Read video: [T, H, W, C], uint8
        # Using read_video from torchvision
        vframes, _, _ = read_video(video_path, output_format="TCHW", pts_unit='sec')
        # read_video might return [T, C, H, W] if output_format="TCHW" is supported?
        # Actually torchvision read_video returns (video, audio, info). 
        # Video is [T, H, W, C] by default. output_format="TCHW" is available in newer versions.
        # Let's check dimensions.
        if vframes.shape[-1] == 3: # [T, H, W, C]
            vframes = vframes.permute(0, 3, 1, 2) # [T, C, H, W]
        
        # Expecting 149 frames
        T = vframes.shape[0]
        
        # 1. Replace first frame with second frame (Index 0 replaced by Index 1)
        if T > 1:
            vframes[0] = vframes[1]
            
        # 2. Pad to 150 frames (Pad last frame)
        target_total = 150
        if T < target_total:
            padding = vframes[-1:].repeat(target_total - T, 1, 1, 1)
            vframes = torch.cat([vframes, padding], dim=0)
        vframes = vframes[:150] # Ensure 150
        
        # 3. Split into Context (0-75) and Target (75-150)
        # First 75 frames -> Context
        # Last 75 frames -> Target
        context_part = vframes[:75]
        target_part = vframes[75:]
        
        # 4. VAE Padding (+1 front, +1 back) -> 77 frames
        def pad_vae(tensor):
            # tensor: [75, C, H, W]
            front = tensor[0:1]
            back = tensor[-1:]
            return torch.cat([front, tensor, back], dim=0)
            
        context_padded = pad_vae(context_part)
        target_padded = pad_vae(target_part)
        
        # Permute to [C, T, H, W] for transform/output
        context_out = context_padded.permute(1, 0, 2, 3)
        target_out = target_padded.permute(1, 0, 2, 3)
        
        return context_out, target_out

    def _process_tensor_transform(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Apply transforms to [C, T, H, W] tensor."""
        # Convert to float [0, 1] if uint8
        if video_tensor.dtype == torch.uint8:
            video_tensor = video_tensor.float() / 255.0
            
        # Resize and CenterCrop
        # We can use v2.functional or the self.transform if it supports batch
        # self.transform typically includes ToTensor (redundant) and Normalize
        # Let's manually apply Resize/Crop/Normalize to be safe and consistent
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        transform = v2.Compose([
            v2.Resize(size=(self.height, self.width), antialias=True),
            v2.CenterCrop(size=(self.height, self.width)),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        video_tensor = transform(video_tensor)

        return video_tensor.permute(1, 0, 2, 3)

    def _load_cameras(self, json_dir: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Load camera JSON, parse, process sync with video.
        Returns: (context_cameras [77, 4, 4], target_cameras [77, 4, 4])
        """
        # Find target json file
        json_files = os.listdir(json_dir)
        target_json = None
        for f in json_files:
            if f.endswith('.json') and '_character.json' not in f and '_check.json' not in f:
                target_json = os.path.join(json_dir, f)
                break
        
        if target_json is None:
            # Return identity matrices if not found
            print(f"Warning: No valid camera JSON found in {json_dir}")
            return [torch.eye(4)] * 77, [torch.eye(4)] * 77
            
        with open(target_json, 'r') as f:
            data = json.load(f)
            
        # Parse all indices
        # keys are strings "0", "1", ...
        # We need indices 0 to 148 (original 149 frames)
        
        # 1. Collect raw matrices
        raw_matrices = []
        for i in range(149):
            key = str(i)
            if key in data:
                mat = _parse_camera_matrix(data[key])
            else:
                # Fallback to previous or identity
                mat = raw_matrices[-1] if raw_matrices else torch.eye(4)
            raw_matrices.append(mat)
            
        # 2. Sync with video: Replace 0 with 1
        if len(raw_matrices) > 1:
            raw_matrices[0] = raw_matrices[1]
            
        # 3. Sync with video: Pad last to 150
        raw_matrices.append(raw_matrices[-1]) # 149 -> 150
        
        # 4. Split
        context_mats = raw_matrices[:75]
        target_mats = raw_matrices[75:]
        
        # 5. VAE Padding
        def pad_mats(mats):
            return [mats[0]] + mats + [mats[-1]]
            
        context_final = pad_mats(context_mats)
        target_final = pad_mats(target_mats)
        
        # Return as list of tensors or stacked tensor
        # Returning list to be compatible with collate or further processing
        # But user mentioned "Context and target camera ... select every 4th". 
        # Dataset returns full sequence, subsampling handled later.
        
        return torch.stack(context_final), torch.stack(target_final)

    def _load_cameras_hemi8(self, json_dir: str, file_name: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Load camera JSON, parse, process sync with video.
        Returns: (context_cameras [77, 4, 4], target_cameras [77, 4, 4])
        """
        # Find target json file
        camera_positions = [
            "C_01_20mm", "C_02_20mm", "C_03_20mm", "C_04_20mm",
            "C_05_20mm", "C_06_20mm", "C_07_20mm", "C_08_20mm"
        ]

        # 暴力查找
        hemi_number = None
        for position in camera_positions:
            if position in file_name:
                hemi_number = position
                break

        if hemi_number is None:
            # Return identity matrices if not found
            print(f"Warning: No valid camera JSON found in {json_dir}")
            assert False

        json_files = os.listdir(json_dir)
        target_json = None
        for f in json_files:
            if f.endswith('.json') and '_character.json' not in f and '_check.json' not in f:
                target_json = os.path.join(json_dir, f)
                break
        
        if target_json is None:
            # Return identity matrices if not found
            print(f"Warning: No valid camera JSON found in {json_dir}")
            return [torch.eye(4)] * 77, [torch.eye(4)] * 77
            
        with open(target_json, 'r') as f:
            data = json.load(f)
            
        # Parse all indices
        # keys are strings "0", "1", ...
        # We need indices 0 to 148 (original 149 frames)
        
        # 1. Collect raw matrices
        raw_matrices = []
        for i in range(149):
            key = str(i)
            if key in data:
                mat = _parse_camera_matrix_hemi8(data[key][hemi_number])
            else:
                # Fallback to previous or identity
                mat = raw_matrices[-1] if raw_matrices else torch.eye(4)
            raw_matrices.append(mat)
            
        # 2. Sync with video: Replace 0 with 1
        if len(raw_matrices) > 1:
            raw_matrices[0] = raw_matrices[1]
            
        # 3. Sync with video: Pad last to 150
        raw_matrices.append(raw_matrices[-1]) # 149 -> 150
        
        # 4. Split
        context_mats = raw_matrices[:75]
        target_mats = raw_matrices[75:]
        
        # 5. VAE Padding
        def pad_mats(mats):
            return [mats[0]] + mats + [mats[-1]]
            
        context_final = pad_mats(context_mats)
        target_final = pad_mats(target_mats)
        
        # Return as list of tensors or stacked tensor
        # Returning list to be compatible with collate or further processing
        # But user mentioned "Context and target camera ... select every 4th". 
        # Dataset returns full sequence, subsampling handled later.
        
        return torch.stack(context_final), torch.stack(target_final)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, list, dict]]:
        video_path, caption, new_name, filename, is_hemi8 = self.samples[index]
        
        # sample_path = os.path.join(self.root_dir, os.path.dirname(video_path))
        # videos_dir = os.path.join(sample_path, "videos")
        json_dir, video_name = os.path.split(video_path)
        json_dir = os.path.dirname(json_dir)
        # try:
        # Load Video
        context_video, target_video = self._load_video_frames(video_path)
        
        # Apply transforms
        context_video = self._process_tensor_transform(context_video)
        target_video = self._process_tensor_transform(target_video)
        
        # Load Cameras
        if not is_hemi8:
            context_cams, target_cams = self._load_cameras(json_dir)
        else:
            context_cams, target_cams = self._load_cameras_hemi8(json_dir, filename)
        
        return {
            "video": target_video,              # [C, 77, H, W]
            "video_camera": target_cams,        # [77, 4, 4]
            "context_frames": context_video,    # [C, 77, H, W]
            "context_camera": context_cams,     # [77, 4, 4]
            "caption": caption,
            "meta": {
                "scene": new_name.split("_")[0],
                "filename": filename,
                "new_name": new_name,
            }
        }


class RefinsDataset(Dataset):
    """
    Dataset for Context-as-Memory-Dataset with MP4 videos.
    
    Structure:
    - root_dir/
        - captions.txt
        - sample_subfolder/
            - videos/
                - video.mp4 (target)
                - videoDepth.mp4
            - json/
                - ...json (camera params)
    
    Process:
    1. Read video (149 frames) -> 0 replaced by 1 -> Pad last to 150.
    2. Split: Context (0-75), Target (75-150).
    3. Pad VAE: Context (+1 front, +1 back) -> 77, Target (+1 front, +1 back) -> 77.
    4. Cameras: Parse JSON, sync with video ops.
    """
    
    def __init__(
        self,
        root_dir: str,
        captions_path: Optional[str] = None,
        height: int = 480,
        width: int = 832, # Updated default to match user request context if needed, kept 480p
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        
        # Use provided captions path or default
        if captions_path is None:
            captions_path = os.path.join(root_dir, "captions.txt")
        self.captions_path = captions_path
        
        self.height = height
        self.width = width
        # Note: transform is applied to the video tensor
        self.transform = transform
        
        self.samples: List[Tuple[str, str, str, str, bool]] = []  # (video_path, caption, new_path, filename, contains_hemi8)
        
        self.resume = True
        self.output_dir = "/m2v_intern_v3/chenkaijin/ref_processed"
        if self.resume:
            self._load_captions_skip_existing()
        else:
            self._load_captions()

        self.rng = random.Random(seed)
        print(f"Loaded {len(self.samples)} samples from captions.txt")
    
    def _load_captions(self):
        """Load and parse captions.txt file."""
        if not os.path.isfile(self.captions_path):
            raise FileNotFoundError(f"Captions file not found: {self.captions_path}")
        
        with open(self.captions_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_caption_line(line)
                if parsed:
                    self.samples.append(parsed)
    
    def _load_captions_skip_existing(self):
        """Load and parse captions.txt file."""
        if not os.path.isfile(self.captions_path):
            raise FileNotFoundError(f"Captions file not found: {self.captions_path}")
        
        with open(self.captions_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_caption_line(line)
                video_path, caption, new_name, filename, is_hemi8 = parsed
                
                scene = new_name.split("_")[0]
                if "Extend" in video_path:
                    output_filename = f"Extend_{scene}_{filename}.tensors.pth"
                else:    
                    output_filename = f"{scene}_{filename}.tensors.pth"
                output_path = os.path.join(self.output_dir, output_filename)
                if os.path.exists(output_path):
                    print(f"File {output_path} already exists, skipping.")
                    continue

                if parsed:
                    self.samples.append(parsed)     

    def _load_video_frames(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video, process frames, split into context and target.
        Returns: (context_frames [C, 77, H, W], target_frames [C, 77, H, W])
        """
        # Read video: [T, H, W, C], uint8
        # Using read_video from torchvision
        vframes, _, _ = read_video(video_path, output_format="TCHW", pts_unit='sec')
        # read_video might return [T, C, H, W] if output_format="TCHW" is supported?
        # Actually torchvision read_video returns (video, audio, info). 
        # Video is [T, H, W, C] by default. output_format="TCHW" is available in newer versions.
        # Let's check dimensions.
        if vframes.shape[-1] == 3: # [T, H, W, C]
            vframes = vframes.permute(0, 3, 1, 2) # [T, C, H, W]
        
        # Expecting 149 frames
        T = vframes.shape[0]
        
        # 1. Replace first frame with second frame (Index 0 replaced by Index 1)
        if T > 1:
            vframes[0] = vframes[1]
            
        # 2. Pad to 150 frames (Pad last frame)
        target_total = 150
        if T < target_total:
            padding = vframes[-1:].repeat(target_total - T, 1, 1, 1)
            vframes = torch.cat([vframes, padding], dim=0)
        vframes = vframes[:150] # Ensure 150
        
        # 3. Split into Context (0-75) and Target (75-150)
        # First 75 frames -> Context
        # Last 75 frames -> Target
        context_part = vframes[:75]
        target_part = vframes[75:]
        
        # 4. VAE Padding (+1 front, +1 back) -> 77 frames
        def pad_vae(tensor):
            # tensor: [75, C, H, W]
            front = tensor[0:1]
            back = tensor[-1:]
            return torch.cat([front, tensor, back], dim=0)
            
        context_padded = pad_vae(context_part)
        target_padded = pad_vae(target_part)
        
        # Permute to [C, T, H, W] for transform/output
        context_out = context_padded.permute(1, 0, 2, 3)
        target_out = target_padded.permute(1, 0, 2, 3)
        
        return context_out, target_out

    def _process_tensor_transform(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Apply transforms to [C, T, H, W] tensor."""
        # Convert to float [0, 1] if uint8
        if video_tensor.dtype == torch.uint8:
            video_tensor = video_tensor.float() / 255.0
            
        # Resize and CenterCrop
        # We can use v2.functional or the self.transform if it supports batch
        # self.transform typically includes ToTensor (redundant) and Normalize
        # Let's manually apply Resize/Crop/Normalize to be safe and consistent
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        transform = v2.Compose([
            v2.Resize(size=(self.height, self.width), antialias=True),
            v2.CenterCrop(size=(self.height, self.width)),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        video_tensor = transform(video_tensor)

        return video_tensor.permute(1, 0, 2, 3)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, list, dict]]:
        video_path, caption, new_name, filename, is_hemi8 = self.samples[index]
        
        context_video, target_video = self._load_video_frames(video_path)
        
        mask_path = video_path.replace(".mp4", "M_CustomDepth.mp4")
        print("提取边界框")
        boxes_origin, boxes_extend, width, height = simple_process(mask_path)

        if len(boxes_origin) > 1:
            boxes_origin[0] = boxes_origin[1]
        
        target_total = 150
        if len(boxes_origin)  < target_total:
            for _ in range(target_total - len(boxes_origin)):
                boxes_origin.append(boxes_origin[-1])
        else:
            boxes_origin = boxes_origin[:target_total]
        
        # 3. 分割为Context和Target
        context_origin = boxes_origin[:75]
        
        # 4. VAE填充
        def pad_frame_list(frame_list):
            """在列表前后各添加一帧"""
            padded = []
            padded.append(frame_list[0])  # 前面
            padded.extend(frame_list)     # 中间
            padded.append(frame_list[-1]) # 后面
            return padded
        
        context_origin = pad_frame_list(context_origin)
        max_square = 0
        max_id = 0
        for i, box in enumerate(context_origin):
            square = (box[2] - box[0]) * (box[3] - box[1])
            max_square = max(max_square, square)
            if square == max_square:
                max_id = i
        
        ref_img = context_video[:, max_id].unsqueeze(1)
        ref_box = context_origin[max_id]

        ref_input = crop_and_center_zoom(ref_img, ref_box)
        ref_input = self._process_tensor_transform(ref_input)
        
        
        
        return {
            "ref": ref_input,              # [C, 77, H, W]
            "meta": {
                "scene": new_name.split("_")[0],
                "filename": filename,
                "new_name": new_name,
                "video_path": video_path,
            }
        }




def build_dataloader_v2(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    captions_path: Optional[str] = None,
    height: int = 480,
    width: int = 832,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Build DataLoader for the new dataset."""
    dataset = ContextMemorySegmentsDataset(
        root_dir=root_dir,
        captions_path=captions_path,
        height=height,
        width=width,
        seed=seed,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
