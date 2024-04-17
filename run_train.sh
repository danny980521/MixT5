#!/usr/bin/bash

#SBATCH -J test-run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g6
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

pwd
which python
hostname

wandb login 4b01cc19aedd9b51e633c61406044403156392ff

python train_t5.py \
--pretrained_model_name_or_path google/t5-v1_1-base \
--save_dir /data/dannykm/repos/SWC/software_capstone/models/t5_v1_1 \
--save_strategy epoch \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 128 \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--use_wandb \
--entity dannykm \
--wandb_model_name t5_v1_1 \
--project_name moe \
# --evaluation_strategy steps \

# python train.py \
# --pretrained_model_name_or_path gogamza/kobart-base-v2 \
# --evaluation_strategy steps \
# --save_dir /data/dannykm/repos/SWC/data_capstone/models/bart_test \
# --save_strategy epoch \
# --per_device_train_batch_size 128 \
# --per_device_eval_batch_size 128 \
# --learning_rate 1e-5 \
# --num_train_epochs 5 \
# --use_wandb \
# --entity dannykm \
# --wandb_model_name kobart_test2 \
# --project_name dialect \
