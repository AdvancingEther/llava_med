#!/bin/bash
    # --lora-path /home/wzc/wzc/aaai26/LLaVA/checkpoints/llava-v1.6-vicuna-7b-slake-finetune_lora \
export CUDA_VISIBLE_DEVICES=6

# python llava/eval/my_model_vqa.py \
#     --model-path /home/wzc/wzc/aaai26/LLaVA/checkpoints/llava-v1.6-vicuna-7b \
#     --lora-path /home/wzc/wzc/aaai26/LLaVA/checkpoints/llava-v1.6-vicuna-7b-vqa_rad-finetune_lora-rank-64 \
#     --image-folder /home/wzc/wzc/aaai26/LLaVA/medical_data/vqa_rad/images \
#     --question-file /home/wzc/wzc/aaai26/LLaVA/medical_data/vqa_rad/questions.jsonl \
#     --answers-file /home/wzc/wzc/aaai26/LLaVA/medical_data/vqa_rad/results/response_lora.jsonl \
#     --conv-mode med_vicuna_v1 \
#     --temperature 0.2 \
#     --num_beams 1 \


python llava/eval/my_model_vqa.py \
    --model-path /home/wzc/wzc/aaai26/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b \
    --lora-path  /home/wzc/wzc/aaai26/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b-vqa_rad-finetune_lora-rank-64\
    --image-folder /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/images \
    --question-file /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/questions.jsonl \
    --answers-file /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/results/response_llava_med_rad_lora.jsonl \
    --conv-mode mistral_instruct \
    --temperature 0.2 \
    --num_beams 1 \


python llava/eval/my_model_vqa.py \
    --model-path /home/wzc/wzc/aaai26/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b \
    --lora-path None \
    --image-folder /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/images \
    --question-file /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/questions.jsonl \
    --answers-file /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/results/response_llava_med_rad_zero_shot.jsonl \
    --conv-mode mistral_instruct \
    --temperature 0.2 \
    --num_beams 1 \

python llava/eval/my_model_vqa.py \
    --model-path /home/wzc/wzc/aaai26/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b-vqa_rad-full-finetune \
    --lora-path None \
    --image-folder /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/images \
    --question-file /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/questions.jsonl \
    --answers-file /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/results/response_llava_med_rad_full_finetune.jsonl \
    --conv-mode mistral_instruct \
    --temperature 0.2 \
    --num_beams 1 \