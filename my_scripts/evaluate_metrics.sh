#!/bin/bash


# python llava/eval/run_eval_metrics.py \
#     --gt /home/wzc/wzc/aaai26/LLaVA/medical_data/slake/questions.jsonl \
#     --pred /home/wzc/wzc/aaai26/LLaVA/medical_data/slake/results/response_llava.jsonl \

# python llava/eval/run_eval_metrics.py \
#     --gt /home/wzc/wzc/aaai26/LLaVA/medical_data/slake/questions.jsonl \
#     --pred /home/wzc/wzc/aaai26/LLaVA/medical_data/slake/results/response_lora.jsonl \

python llava/eval/run_eval_metrics.py \
    --gt /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/questions.jsonl \
    --pred /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/results/response_llava_med_rad_full_finetune.jsonl \

python llava/eval/run_eval_metrics.py \
    --gt /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/questions.jsonl \
    --pred /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/results/response_llava_med_rad_lora.jsonl \

python llava/eval/run_eval_metrics.py \
    --gt /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/questions.jsonl \
    --pred /home/wzc/wzc/aaai26/LLaVA-Med/medical_data/vqa_rad/results/response_llava_med_rad_zero_shot.jsonl \