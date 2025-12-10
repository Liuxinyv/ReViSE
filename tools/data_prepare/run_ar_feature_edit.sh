#!/bin/bash
BASE_DIR=$(pwd)
export PYTHONPATH="$BASE_DIR:$BASE_DIR/nets/third_party:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="0"
DATA_FILE="processed/vae_feature"
OUTPUT_DIR="processed/ar_feature"

python -m torch.distributed.launch \
    --nnodes=1 \
    --nproc_per_node=1 \
    --use_env \
    --master_port 1234 \
    tools/data_prepare/ar_feature_extract_edit.py \
    --model-path revise_ckpt/ar_model/checkpoint \
    --data-file ${DATA_FILE} \
    --result-folder ${OUTPUT_DIR}