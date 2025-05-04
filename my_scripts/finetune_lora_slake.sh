#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# --version $PROMPT_VERSION \
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

#  
export LOCAL_RANK=1
export MASTER_PORT=26002

deepspeed --master_port 26002 --include localhost:5 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --version "mistral_instruct" \
    --local_rank $LOCAL_RANK \
    --lora_enable True \
    --model_name_or_path /home/wzc/wzc/aaai26/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b \
    --data_path /home/wzc/wzc/aaai26/LLaVA/medical_data/slake/conversations.json \
    --image_folder /home/wzc/wzc/aaai26/LLaVA/medical_data/slake/images \
    --vision_tower /home/wzc/wzc/aaai26/LLaVA/checkpoints/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-med-v1.5-mistral-7b-slake-full-finetune-15epochs \
    --num_train_epochs 15 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --wandb_run_name "slake-finetune-lora-15epochs" \
    --wandb_project "llava-med-slake-finetuned"
