set -x

export DEBUG_MODE="true"
VERSION="0.1"
RUN_NAME="PresAesth_v${VERSION}"
export LOG_PATH="log/debug_${RUN_NAME}_${VERSION}.txt"

# Envs
export OMP_NUM_THREADS=8
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source train_env/bin/activate
uv run torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 \
    src/train_multi_task.py \
    --output_dir output/${RUN_NAME} \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --max_prompt_length 4096 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 200 \
    --score_reward_threshold 0.3 \
    --beta 0.001 \
    --deepspeed local_scripts/zero2.json \
    --dataset_config data_config/train_dataset.yaml \
    --deficiency_f1_threshold 0.6 \
    --reward_funcs accuracy format comparison \