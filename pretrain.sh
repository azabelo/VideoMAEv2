#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=$((12000 + $RANDOM % 20000))  # Randomly set master_port to avoid port conflicts
export OMP_NUM_THREADS=1  # Control the number of threads

OUTPUT_DIR='YOUR_PATH/work_dir/vit_g_hybrid_pt_1200e'  # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='official_hmdb_splits1/train.csv'  # The data list file path.
# pretrain data list file follows the following format
# for the video data line: video_path, 0, -1
# for the rawframe data line: frame_folder_path, start_index, total_frames

JOB_NAME=$1  # the job name of the slurm task
PARTITION=${PARTITION:-"video"}  # Name of the partition
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-64}  # Number of GPUs
GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # Number of GPUs in each node
CPUS_PER_TASK=${CPUS_PER_TASK:-14}  # Number of CPU cores allocated, number of tasks equal to the number of GPUs used
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:2}  # Other training args

# Please refer to `run_mae_pretraining.py` for the meaning of the following hyperreferences
        python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model pretrain_videomae_giant_patch14_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --with_checkpoint \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 4 \
        --num_workers 10 \
        --opt adamw \
        --lr 6e-4 \
        --clip_grad 0.02 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 30 \
        --save_ckpt_freq 5 \
        --epochs 300 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}
