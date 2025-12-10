#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import pickle as pkl
warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import sys
sys.path.append("/aifs4su/liuxinyu/ReViSE/")
from nets.revise.revise_video_generator import ReviseVideoGenerator
from nets.third_party.wan.configs import SIZE_CONFIGS
from nets.third_party.wan.utils.utils import cache_video, cache_image, str2bool
from pathlib import Path

def _get_rank_env():
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device

def _make_save_file(output_dir: str, prompt: str, src_file_path: str, idx: str) -> str:
    """
    Constructs a save file path based on a predefined naming convention.
    """
    source_filename_path = Path(src_file_path)
    base_name = source_filename_path.stem
    ext = source_filename_path.suffix     
    indexed_filename = f"{base_name}{idx}{ext}"
    filename = f"{indexed_filename}"
    
    return str(Path(output_dir) / filename)

def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args: 
        args: Parsed command line arguments
        
    Raises:
        ValueError: If validation fails
    """
    if not args.prompt:
        raise ValueError("Please provide a prompt using --prompt argument")
        
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None: 
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832", "352*640", "640*352"]:
            args.sample_shift = 3.0

    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I validation
    if "t2i" in args.task and args.frame_num != 1:
        raise ValueError(f"Frame number must be 1 for t2i task, got {args.frame_num}")

    # Set random seed if not provided
    if args.base_seed < 0:
        args.base_seed = torch.randint(0, 2**32, (1,)).item()


def str2tuple(v: str) -> tuple:
    """
    Convert string to tuple.
    
    Examples:
        '1,2,2' -> (1, 2, 2)
        '(1,2,2)' -> (1, 2, 2)
        
    Args:
        v: String representation of a tuple
        
    Returns:
        Parsed tuple with integer values
    """
    v = v.strip()
    if v.startswith('(') and v.endswith(')'):
        v = v[1:-1]
    
    return tuple(int(x.strip()) for x in v.split(','))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ReviseVideo: Unified Video Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument(
        "--task", type=str, default="t2v",
        choices=["t2v", "t2i", "i2i", "v2v"],
        help="Generation task type"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--size", type=str, default="832*480",
        help="Output size in format 'width*height'"
    )
    parser.add_argument(
        "--frame_num", type=int, default=None,
        help="Number of frames to generate (should be 4n+1 for videos)"
    )
    parser.add_argument(
        "--sample_fps", type=int, default=8,
        help="FPS of the generated video"
    )
    
    # Model paths (now fixed)
    parser.add_argument(
        "--ckpt_dir", type=str,
        help="Path to the main model checkpoint directory (now fixed)"
    )
    parser.add_argument(
        "--adapter_ckpt_dir", type=str,
        help="Path to the adapter checkpoint (now fixed)"
    )
    parser.add_argument(
        "--vision_head_ckpt_dir", type=str,
        help="Path to the vision head checkpoint (now fixed)"
    )
    parser.add_argument(
        "--new_checkpoint", type=str,
        help="Path to additional checkpoint to load (now fixed)"
    )
    parser.add_argument(
        "--ar_model_path", type=str,
        help="Path to the AR model (now fixed)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--sample_solver", type=str, default='unipc',
        choices=['unipc', 'dpm++'],
        help="Sampling solver"
    )
    parser.add_argument(
        "--sample_steps", type=int, default=None,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--sample_shift", type=float, default=None,
        help="Sampling shift factor"
    )
    parser.add_argument(
        "--sample_guide_scale", type=float, default=5.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--base_seed", type=int, default=-1,
        help="Random seed (-1 for random)"
    )
    
    # Input files
    parser.add_argument(
        "--src_file_path", type=str, default=None,
        help="Source image/video path for editing tasks"
    )
    parser.add_argument(
        "--save_file", type=str,
        help="Output file path (auto-generated if not specified)"
    )
    
    # Advanced parameters
    parser.add_argument(
        "--adapter_in_channels", type=int, default=1152,
        help="Adapter input channels"
    )
    parser.add_argument(
        "--adapter_out_channels", type=int, default=4096,
        help="Adapter output channels"
    )
    parser.add_argument(
        "--adapter_query_length", type=int, default=256,
        help="Adapter query length"
    )
    parser.add_argument(
        "--use_visual_context_adapter", type=str2bool, default=False,
        help="Whether to use visual context adapter"
    )
    parser.add_argument(
        "--visual_context_adapter_patch_size", type=str2tuple, default=(1, 4, 4),
        help="Visual context adapter patch size (e.g., '1,4,4')"
    )
    parser.add_argument(
        "--use_visual_as_input", type=str2bool, default=False,
        help="Whether to use visual as input"
    )
    parser.add_argument(
        "--condition_mode", type=str, default="full",
        help="Conditioning mode"
    )
    parser.add_argument(
        "--max_context_len", type=int, default=1024,
        help="Maximum context length"
    )
    
    # Classifier-free guidance
    parser.add_argument(
        "--classifier_free_ratio", type=float, default=0.0,
        help="Classifier-free guidance ratio"
    )
    parser.add_argument(
        "--unconditioned_context_path", type=str,
        help="Path to unconditioned context embeddings (now fixed)"
    )
    parser.add_argument(
        "--unconditioned_context_length", type=int, default=256,
        help="Unconditioned context length"
    )
    parser.add_argument(
        "--special_tokens_path", type=str,
        help="Path to special tokens file (now fixed)"
    )
    
    # AR model parameters
    parser.add_argument(
        "--ar_model_num_video_frames", type=int, default=8,
        help="Number of video frames for AR model"
    )
    parser.add_argument(
        "--ar_query", type=str,
        help="Query for AR model"
    )
    parser.add_argument(
        "--ar_conv_mode", type=str, default="llama_3",
        help="AR model conversation mode"
    )
    parser.add_argument(
        "--output_dir", type=str, default='./output',
        help="Number of frames to skip"
    )
    # Video processing
    parser.add_argument(
        "--sampling_rate", type=int, default=3,
        help="Video sampling rate"
    )
    parser.add_argument(
        "--skip_num", type=int, default=0,
        help="Number of frames to skip"
    )
    parser.add_argument(
        "--prompts_file", type=str, default='/aifs4su/liuxinyu/HOIGen/single_motion.json',
        help="Number of frames to skip"
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def init_logging(rank: int) -> None:
    """Initialize logging configuration."""
    log_file = f'revisevideo_generate_rank{rank}.log'
    
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def transform_image_to_tensor(image: Union[Image.Image, np.ndarray], 
                            target_size: Tuple[int, int] = (480, 832)) -> torch.Tensor:
    """
    Transform PIL Image or numpy array to tensor with resize and center crop.
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        
    Returns:
        Transformed tensor
    """
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size
        
    ratio = float(target_size[1]) / float(target_size[0])  # w/h
    
    if w < h * ratio:
        crop_size = (int(float(w) / ratio), w)
    else:
        crop_size = (h, int(float(h) * ratio))

    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    return transform(image)


def extract_vae_features(image_path: str, vae, device: torch.device, 
                        target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    """
    Extract VAE features from image.
    
    Args:
        image_path: Path to image file
        vae: VAE model instance
        device: Computation device
        target_size: Target image size
        
    Returns:
        VAE encoded features or None if failed
    """
    if not os.path.exists(image_path):
        logging.warning(f"Image file not found: {image_path}")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform_image_to_tensor(image, target_size)
        image_tensor = image_tensor.unsqueeze(1).to(device)  # [C, 1, H, W]

        with torch.no_grad():
            latent_feature = vae.encode([image_tensor])
            latent_feature = latent_feature[0]
        
        return latent_feature
        
    except Exception as e:
        logging.error(f"Failed to extract VAE features from {image_path}: {e}")
        return None


def read_video_frames(video_path: str, frame_num: int, sampling_rate: int = 3, 
                     skip_num: int = 0, target_size: Tuple[int, int] = (480, 832)) -> Optional[torch.Tensor]:
    """
    Read video frames and convert to tensor.
    
    Args:
        video_path: Path to video file
        frame_num: Number of frames to extract
        sampling_rate: Frame sampling rate
        skip_num: Number of frames to skip at beginning
        target_size: Target frame size (height, width)
        
    Returns:
        Frame tensor [T, C, H, W] or None if failed
    """
    if not os.path.exists(video_path):
        logging.warning(f"Video file not found: {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return None
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logging.info(f"Video info: frames={total_frames}, fps={fps}, size={width}x{height}")

        # Adjust sampling rate if needed
        while total_frames < frame_num * sampling_rate + skip_num:
            sampling_rate -= 1
            if sampling_rate == 0:
                logging.warning(f"Cannot extract {frame_num} frames from video")
                return None
                
        logging.info(f"Using sampling rate: {sampling_rate}")

        # Check aspect ratio compatibility
        target_aspect = target_size[1] / target_size[0]  # w/h
        video_aspect = width / height
        
        if abs(target_aspect - video_aspect) > 0.5:  # Significant aspect ratio difference
            logging.warning(f"Aspect ratio mismatch: target={target_aspect:.2f}, video={video_aspect:.2f}")
        
        # Extract frames
        frames = []
        current_frame = 0
        
        while current_frame < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame < skip_num:
                current_frame += 1
                continue
            
            if (current_frame - skip_num) % sampling_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
            current_frame += 1
            
            if len(frames) >= frame_num:
                break
        
        if len(frames) != frame_num:
            logging.warning(f"Extracted {len(frames)} frames, expected {frame_num}")
            return None
        
        # Convert to tensor
        frame_tensors = []
        for frame in frames:
            frame_tensor = transform_image_to_tensor(Image.fromarray(frame), target_size)
            frame_tensors.append(frame_tensor)
        
        return torch.stack(frame_tensors)  # [T, C, H, W]
        
    except Exception as e:
        logging.error(f"Failed to read video frames: {e}")
        return None
        
    finally:
        cap.release()

def main():
    """Main function for single video generation from --prompt and --src_file_path."""
    args = parse_args()

    rank, world_size, local_rank, device = _get_rank_env()
    init_logging(rank)

    try:
        # 检查必须参数
        if args.src_file_path is None:
            raise ValueError("Please provide --src_file_path for i2i/v2v/t2v-from-video tasks.")
        if not os.path.exists(args.src_file_path):
            raise FileNotFoundError(f"Source file not found: {args.src_file_path}")

        # 构建生成器（内部会 init_process_group 一次）
        generator = ReviseVideoGenerator(args)
        generator.setup_distributed()

        # 同步随机种子
        if dist.is_initialized():
            base_seed = [args.base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = int(base_seed[0]) + rank * 1000
        if args.base_seed < 0:
            args.base_seed = torch.randint(0, 2**32, (1,)).item() + rank * 1000

        torch.manual_seed(args.base_seed)
        np.random.seed(args.base_seed % (2**32 - 1))

        # 加载模型
        generator.load_special_tokens()
        generator.load_unconditioned_context()
        generator.initialize_models()

        # 输出目录
        output_dir = args.output_dir
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # 解析 size
        if args.size in SIZE_CONFIGS:
            target_size = SIZE_CONFIGS[args.size]
        else:
            try:
                w, h = args.size.split('*')
                target_size = (int(h), int(w))  # (H, W)
            except Exception:
                target_size = (480, 832)
                logging.warning(f"Invalid size format: {args.size}, using default {target_size}")
        logging.info(f"[rank{rank}] target_size={target_size}, world_size={world_size}")

        # ===== 仅处理一个任务 =====
        prompt = args.prompt
        src_file_path = args.src_file_path

        # 决定保存路径
        if args.save_file is not None:
            save_file = args.save_file
        else:
            # 如果没有 idx 概念，传一个固定字符串，比如 "0"
            save_file = _make_save_file(output_dir, prompt, src_file_path, idx="0")

        # 再检查一次（稳妥）
        if not os.path.exists(src_file_path):
            logging.warning(f"[rank{rank}] missing src: {src_file_path}")
            return
        if os.path.exists(save_file) and rank == 0:
            logging.info(f"[rank{rank}] exists, will overwrite: {save_file}")

        # AR 模型推理（决定任务类型）
        ar_query = args.ar_query or "<video>\n Describe this video and its style in a very detailed manner."
        ar_caption_ids, ar_caption = generator.ar_model.generate(src_file_path, prompt)
        gen_mode_1 = torch.any(ar_caption_ids == 128003)
        gen_mode_2 = torch.any(ar_caption_ids == 128002)

        if not gen_mode_1 and not gen_mode_2:
            logging.info(f"[rank{rank}] understanding only, skip. output: {ar_caption}")
            return

        # 根据源文件类型和 AR 模式判断 task
        if src_file_path.endswith(('.png', '.jpg', '.jpeg')):
            task = 'i2i'
        elif src_file_path.endswith('.mp4'):
            task = 'v2v'
        else:
            task = 't2v' if gen_mode_1 else ('t2i' if gen_mode_2 else None)

        if task is None:
            logging.warning(f"[rank{rank}] cannot determine task for {src_file_path}")
            return

        if task in ['t2i', 'i2i']:
            args.frame_num = 1
        else:
            if args.frame_num == 1:
                args.frame_num = 81

        visual_emb = None
        if task == 'i2i':
            visual_emb = extract_vae_features(
                src_file_path, generator.revisevideo_x2x.vae,
                generator.device, (target_size[1], target_size[0])
            )
            if visual_emb is not None:
                visual_emb = visual_emb[:, 0:1]
        elif task == 'v2v':
            frames_tensor = read_video_frames(
                src_file_path, args.frame_num, args.sampling_rate,
                args.skip_num, (target_size[1], target_size[0])
            )
            if frames_tensor is None:
                logging.warning(f"[rank{rank}] insufficient/unreadable frames: {src_file_path}")
                return
            frames_tensor = frames_tensor.to(generator.device)
            with torch.no_grad():
                visual_emb = generator.revisevideo_x2x.vae.encode(
                    frames_tensor.transpose(0, 1).unsqueeze(0)
                )[0]

        vlm_last_hidden_states = generator.ar_model.general_emb(
            prompt, src_file_path, task_type=task
        )

        logging.info(
            f"[rank{rank}] task={task} ar_emb={tuple(vlm_last_hidden_states.shape)} "
            f"visual={None if visual_emb is None else tuple(visual_emb.shape)} -> {save_file}"
        )

        result = generator.revisevideo_x2x.generate(
            prompt,
            visual_emb=visual_emb,
            ar_vision_input=vlm_last_hidden_states,
            size=(target_size[0], target_size[1]),
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            special_tokens=generator.special_tokens,
            classifier_free_ratio=args.classifier_free_ratio,
            unconditioned_context=generator.unconditioned_context,
            condition_mode=args.condition_mode,
            use_visual_as_input=args.use_visual_as_input,
        )

        if result is None:
            logging.warning(f"[rank{rank}] generation returned None")
            return

        if rank == 0:
            if args.frame_num == 1:
                cache_image(
                    tensor=result.squeeze(1)[None],
                    save_file=save_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
            else:
                cache_video(
                    tensor=result[None],
                    save_file=save_file,
                    fps=args.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
            logging.info(f"[rank{rank}] saved -> {save_file}")

        if dist.is_initialized():
            dist.barrier()

    except Exception as e:
        logging.error(f"[rank{rank}] Generation failed: {e}", exc_info=True)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()




if __name__ == "__main__":
    main()