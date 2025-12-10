BASE_DIR=$(pwd)
export PYTHONPATH="$BASE_DIR:$BASE_DIR/nets/third_party:${PYTHONPATH}"

CKPT_DIR="revise_ckpt/wan/wanxiang1_3b"

VIDEO_LIST_PATH="training_metadata/local_filtered_rewrite.json"
OUTPUT_DIR="processed/vae_feature"

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=$NGPUS_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  /aifs4su/liuxinyu/Omni-Video/tools/data_prepare/vae_feature_extract_edit.py \
  --task "t2v-1.3B" \
  --ckpt_dir ${CKPT_DIR} \
  --video_list_path ${VIDEO_LIST_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --frame_num 17 \
  --sampling_rate 2 \
  --skip_num 0 \
  --target_size "352,640"
