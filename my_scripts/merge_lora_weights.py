import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from peft import PeftModel

def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name, device_map='cpu')
    lora_model = PeftModel.from_pretrained(model, args.lora_path)
    model = lora_model.merge_and_unload()


    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b")
    parser.add_argument("--lora-path", type=str, default="/root/autodl-tmp/LLaVA-Med/checkpoints/llava-slake-finetuned")
    parser.add_argument("--save-model-path", type=str, default="/root/autodl-tmp/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b-lora")

    args = parser.parse_args()

    merge_lora(args)