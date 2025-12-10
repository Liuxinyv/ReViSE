#!/bin/bash
export MASTER_ADDR=127.0.0.1
export MAIN_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

export NNODES=1
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 \
    --master_addr "$MASTER_ADDR" --master_port "$MAIN_PORT" \
    tools/inference/generative.py \
    --sample_solver unipc \
    --adapter_in_channels 1152 \
    --adapter_out_channels 4096 \
    --adapter_query_length 256 \
    --use_visual_context_adapter true \
    --visual_context_adapter_patch_size 1,4,4 \
    --use_visual_as_input false \
    --condition_mode full \
    --max_context_len 2560 \
    --ar_model_num_video_frames 8 \
    --ar_conv_mode llama_3 \
    --sampling_rate 1 \
    --skip_num 1 \
    --unconditioned_context_length 2560 \
    --classifier_free_ratio 0.0 \
    --size 640*352 \
    --frame_num 17 \
    --sample_fps 8 \
    --sample_steps 40 \
    --sample_guide_scale 3.0 \
    --sampling_rate 1 \
    --base_seed 1818 \
    --prompt "What if the the dog ran into the depth of a forest?" \
    --src_file_path ./examples/source_0.mp4 \
    --output_dir output/

torchrun --nproc_per_node=1 \
    --master_addr "$MASTER_ADDR" --master_port "$MAIN_PORT" \
    tools/inference/generative.py \
    --sample_solver unipc \
    --adapter_in_channels 1152 \
    --adapter_out_channels 4096 \
    --adapter_query_length 256 \
    --use_visual_context_adapter true \
    --visual_context_adapter_patch_size 1,4,4 \
    --use_visual_as_input false \
    --condition_mode full \
    --max_context_len 2560 \
    --ar_model_num_video_frames 8 \
    --ar_conv_mode llama_3 \
    --sampling_rate 1 \
    --skip_num 1 \
    --unconditioned_context_length 2560 \
    --classifier_free_ratio 0.0 \
    --size 640*352 \
    --frame_num 17 \
    --sample_fps 8 \
    --sample_steps 40 \
    --sample_guide_scale 3.0 \
    --sampling_rate 1 \
    --base_seed 1818 \
    --prompt "What if the scene transitioned from a magical night to a dawn, causing the northern lights to fade away?" \
    --src_file_path examples/source_1.mp4 \
    --output_dir output/

torchrun --nproc_per_node=1 \
    --master_addr "$MASTER_ADDR" --master_port "$MAIN_PORT" \
    tools/inference/generative.py \
    --sample_solver unipc \
    --adapter_in_channels 1152 \
    --adapter_out_channels 4096 \
    --adapter_query_length 256 \
    --use_visual_context_adapter true \
    --visual_context_adapter_patch_size 1,4,4 \
    --use_visual_as_input false \
    --condition_mode full \
    --max_context_len 2560 \
    --ar_model_num_video_frames 8 \
    --ar_conv_mode llama_3 \
    --sampling_rate 1 \
    --skip_num 1 \
    --unconditioned_context_length 2560 \
    --classifier_free_ratio 0.0 \
    --size 640*352 \
    --frame_num 17 \
    --sample_fps 8 \
    --sample_steps 40 \
    --sample_guide_scale 3.0 \
    --sampling_rate 1 \
    --base_seed 1818 \
    --prompt "What if the girl's fragrance gently attracted a delicate butterfly, fluttering toward her?" \
    --src_file_path examples/source_2.mp4 \
    --output_dir output/