import argparse
import torch
import os
from llava import conversation as conversation_lib
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
from transformers import set_seed, logging
from peft import PeftModel

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def parse_args():
    parser = argparse.ArgumentParser(description='Medical Image VQA Inference')
    parser.add_argument("--model-path", type=str, default="/home/wzc/wzc/aaai26/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b",
                       help='Path to LLaVA-Med model')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--lora-path", type=str, default="/home/wzc/wzc/aaai26/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b-vqa_rad-finetune_lora-rank-64")
    return parser.parse_args()

def init_llava_model(model_path,model_base):
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _= load_pretrained_model(model_path, model_base, model_name=model_name)

    return tokenizer,model,image_processor


def process_special_tokens(text, model_config):
    """处理特殊token"""
    # 移除已有的特殊token
    text = text.replace(DEFAULT_IMAGE_TOKEN, '').strip()
    
    # 根据模型配置添加特殊token
    if model_config.mm_use_im_start_end:
        text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
    else:
        text = DEFAULT_IMAGE_TOKEN + '\n' + text
        
    # 处理其他特殊token
    if hasattr(model_config, 'extra_tokens'):
        for token in model_config.extra_tokens:
            if token in text:
                pass
                
    return text

def basic_vqa(text_input,image_path, model, tokenizer,image_processor):
    # 添加输入验证
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

    # 处理图像token
    if model.config.mm_use_im_start_end:
        text_input = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text_input
    else:
        text_input = DEFAULT_IMAGE_TOKEN + '\n' + text_input


    conv = conv_templates["mistral_instruct"].copy()

    conv.append_message(conv.roles[0], text_input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 根据不同的分隔符风格处理
    if conv.sep_style == SeparatorStyle.TWO:
        sep = conv.sep2
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        prompt = conv.get_prompt() + conv.sep
    elif conv.sep_style == SeparatorStyle.MPT:
        sep = conv.sep + conv.roles[1]

    elif conv.sep_style == SeparatorStyle.LLAMA_2:
        sep = "[/INST]"

    else:
        sep = conv.sep

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

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
    
    # 确保数据类型一致
    dtype = torch.float16 if model.config.torch_dtype == torch.float16 else torch.float32
    
    # 图像处理
    image_tensor = image_tensor.to(dtype=dtype, device=model.device)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            max_new_tokens=128,
            temperature=0.2,
            num_beams=1,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    print(output_ids)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs




def main():
    args = parse_args()
    tokenizer, model, image_processor = init_llava_model(args.model_path, args.model_base)

    # test_text = "Hello,please introduce yourself"
    # print("User:", test_text)
    # response = basic_vqa(test_text, model, tokenizer,image_processor)
    # print("Assistant:", response)
    lora_model = PeftModel.from_pretrained(model, args.lora_path)
    lora_model.eval()

    text_input = "What modality is used to take this image?"
    image_path = "/home/wzc/wzc/aaai26/LLaVA-Med/medical_data/slake/images/xmlab0/source.jpg"
    response = basic_vqa(text_input,image_path,lora_model,tokenizer,image_processor)
    print(response)

if __name__ == "__main__":
    main()