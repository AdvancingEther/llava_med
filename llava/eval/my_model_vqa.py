import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from peft import PeftModel

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def init_llava_model(model_path, model_base):

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    return tokenizer, model, image_processor, context_len


def eval_model(args):
    set_seed(0)
    
    # 初始化模型
    tokenizer, model, image_processor, context_len = init_llava_model(args.model_path, args.model_base)
    
    # 加载LoRA权重（如果指定）
    if args.lora_path is not None and args.lora_path.lower() != "none":
        print(f"Loading LoRA weights from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
    
    model.eval()

    # 读取问题数据
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # 准备输出文件
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["question"]
        
        # 处理问题文本
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # 准备对话模板
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 处理输入
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # 处理图像
        image_path = os.path.join(args.image_folder, image_file)
        
        image = Image.open(image_path)
        image = image.convert('RGB')  # 确保图像是RGB格式
        if model.config.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # 设置停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        # 生成回答
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=64,
                use_cache=True,
                cache_position=None)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # 保存结果
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": qs.replace(DEFAULT_IMAGE_TOKEN, '').strip(),
            "response": outputs,
            "answer_id": ans_id,
            "model_id": get_model_name_from_path(args.model_path),
            "metadata": {}
        }) + "\n")
        ans_file.flush()
    
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/wzc/wzc/aaai26/LLaVA/checkpoints/llava-v1.6-vicuna-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/wzc/wzc/aaai26/LLaVA/medical_data/slake/images")
    parser.add_argument("--question-file", type=str, default="/home/wzc/wzc/aaai26/LLaVA/medical_data/slake/questions.jsonl")
    parser.add_argument("--answers-file", type=str, default="/home/wzc/wzc/aaai26/LLaVA/medical_data/slake/results/response_lora.jsonl")
    parser.add_argument("--conv-mode", type=str, default="med_vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--lora-path", type=str, default="/home/wzc/wzc/aaai26/LLaVA/checkpoints/llava-v1.6-vicuna-7b-slake-finetune_lora", help="Path to LoRA weights to be merged with the base model")
    args = parser.parse_args()

    eval_model(args)
