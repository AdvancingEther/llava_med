#!/bin/bash

#  
export LOCAL_RANK=1
export MASTER_PORT=26001

deepspeed --master_port 26001 --include localhost:1,2,3,4,5,6 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --version "mistral_instruct" \
    --local_rank $LOCAL_RANK \
    --lora_enable False \
    --freeze_backbone False \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter False \
    --model_name_or_path /home/wzc/wzc/aaai26/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b \
    --data_path /home/wzc/wzc/aaai26/LLaVA/medical_data/vqa_rad/train_data/conversations.json \
    --image_folder /home/wzc/wzc/aaai26/LLaVA/medical_data/vqa_rad/train_data/images \
    --vision_tower /home/wzc/wzc/aaai26/LLaVA/checkpoints/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-med-v1.5-mistral-7b-vqa_rad-full-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
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
    --dataloader_num_workers 1 \
    --report_to wandb
