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
from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict, save_video
from diffsynth.models.wan_video_dit import SelfAttention
import torchvision
from PIL import Image
import numpy as np
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import shutil
from vae_process_dataset import ContextMemorySegmentsDataset, RefinsDataset
from typing import Optional, Dict, Any, List


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
            is_extend = ("Extend" in meta["new_name"][i])
            # For new dataset, we might use filename or other ID
            if "filename" in meta:
                filename_base = meta["filename"][i]
                if is_extend:
                    output_filename = f"Extend_{scene}_{filename_base}.tensors.pth"
                else:
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



class LightningModelForRefine(pl.LightningModule):
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
        
        model_path = [vae_path]
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

        ref_input = batch["ref"]
        meta = batch["meta"]         # Batch metadata
        
        batch_size = ref_input.shape[0]
        
        # Process each item in the batch
        for i in range(batch_size):
            # Create output filename based on scene and frame range
            scene = meta["scene"][i]
            is_extend = ("Extend" in meta["video_path"][i])
            # For new dataset, we might use filename or other ID
            if "filename" in meta:
                filename_base = meta["filename"][i]
                if is_extend:
                    output_filename = f"Extend_{scene}_{filename_base}.tensors.pth"
                else:
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
                video_latents = self.encode_video_batch(ref_input[i:i+1])  # Keep batch dimension
                video_latents = video_latents.squeeze(0)  # Remove batch dimension [C, 20, H, W]
                
                # vis_video = self.pipe.decode_video(video_latents.unsqueeze(0), **self.tiler_kwargs)[0]
                # vis_video = self.pipe.tensor2video(vis_video)
                # save_video(vis_video, os.path.join("results", f"ref_check_video{batch_idx}.mp4"), fps=15, quality=5)
                # Prepare output data
                output_data = {
                    # Video latents
                    "ref_latents": video_latents,           # [C_latent, 20, H_latent, W_latent]
                }
                
                # Save to file
                torch.save(output_data, output_path)
                print(f"Saved: {output_path}")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}, item {i}: {e}")
                continue



def parse_args():
    parser = argparse.ArgumentParser(description="Train ReCamMaster")
    parser.add_argument(
        "--task",
        type=str,
        default="refine",
        # required=True,
        choices=["data_process", "refine"],
        help="Task. `data_process` or `refine`.",
    )
    parser.add_argument(
        "--captions_path",
        type=str,
        default="/m2v_intern_v3/chenkaijin/final_train.txt",
        help="Path of captions file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/m2v_intern/chenkaijin/single_processed",
        # required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/m2v_intern_v3/chenkaijin/ref_processed",
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
        default=200,
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
        default=0,
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
        default=False,
        help="Whether to modify rope",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = ContextMemorySegmentsDataset(
        root_dir=args.dataset_path,
        captions_path=args.captions_path,
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
    
    
def refine(args):
    dataset = RefinsDataset(
        root_dir=args.dataset_path,
        captions_path=args.captions_path,
        height=args.height,
        width=args.width,
        seed=42
    )
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # Process in order
        num_workers=0,   #args.dataloader_num_workers,
        pin_memory=False,
        drop_last=False

    )
    model = LightningModelForRefine(
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


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    if args.task == "data_process":
        data_process(args)
    elif args.task == "refine":
        refine(args)
