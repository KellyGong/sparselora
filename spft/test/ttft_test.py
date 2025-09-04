import argparse
import json
import os
import re
import time

import torch
from peft import AutoPeftModelForCausalLM
from tabulate import tabulate
from tqdm import trange
from collections import defaultdict

from spft.utils import distributed as dist
from spft.api import SPFTConfig, load_channel_act_file, get_spft_model
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM, AutoModel


def safe_load(model_dir):
    from safetensors import torch as safetorch

    state_dict = {}
    for file in os.listdir(model_dir):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_dir, file)
            state_dict.update(safetorch.load_file(file_path))

    return state_dict


def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def extract_answer(response: str, dataset) -> str | float:
    response = response.strip().lower()
    print(response)
    if dataset == "boolq":
        answers = re.findall(r"true|false", response)
        if not answers:
            return ""
        return answers[0]
    elif dataset == "piqa":
        answers = re.findall(r"1|2", response)
        if not answers:
            return ""
        return "solution" + answers[0]
    elif dataset in ["social-iqa", "arc-challenge", "arc-easy", "openbookqa"]:
        answers = re.findall(r"1|2|3|4|5", response)
        if not answers:
            return ""
        return "answer" + answers[0]
    elif dataset == "hellaswag":
        answers = re.findall(r"1|2|3|4", response)
        if not answers:
            return ""
        return "ending" + answers[0]
    elif dataset == "winogrande":
        answers = re.findall(r"1|2", response)
        if not answers:
            return ""
        return "option" + answers[0]
    elif dataset in ["gsm8k", "mawps", "svamp"]:
        response = response.replace(',', '')
        answers = [s for s in re.findall(r'-?\d+\.?\d*', response)]
        if not answers:
            return float('inf')
        return float(answers[-1])
    else:
        raise ValueError(f"Unsupported dataset: '{dataset}'")


def main(args):
    
    if "gsm8k" in args.dataset or "mawps" in args.dataset or "svamp" in args.dataset:
        args.max_new_tokens = 256

    # dist.init()
    # devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    # torch.cuda.set_device(devices[0])
    # max_memory = {device: torch.cuda.get_device_properties(device).total_memory for device in devices}

    spft_config = SPFTConfig.from_file(os.path.join(args.model_name_or_path, 'args.json'))

    if spft_config and spft_config.peft != 'reft':
        model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        # max_memory=max_memory,
        ).to('cuda')
        tokenizer_path = model.peft_config["default"].base_model_name_or_path

    elif spft_config:
        model = AutoModelForCausalLM.from_pretrained(
            spft_config.model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to('cuda')
        tokenizer_path = spft_config.model_id
    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).to('cuda')
        tokenizer_path = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=512,
        padding_side="left",
        use_fast=False,
    )

    if len(tokenizer) > 32000: #* Llama3
        print("Using LLaMA 3 tokenizer")
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})


    if spft_config and spft_config.peft == 'reft':

        spft_config.padding_side = tokenizer.padding_side
    
        model = get_spft_model(model, tokenizer, spft_config, 
                               enable_static=args.enable_static, reft=spft_config.peft,
                               enable_unsloth=False).to('cuda')
    
        state_dict = torch.load(os.path.join(args.model_name_or_path, "reft.pth"), map_location="cuda", weights_only=True)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"Unexpected keys: {unexpected_keys}")

    model = model.to(torch.bfloat16)

    generation_config = GenerationConfig(
        max_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    metrics = {}
    times = defaultdict(list)
    results = defaultdict(list)
    for dataset in args.dataset.split("+") if "+" in args.dataset else [args.dataset]:
        eval_path = os.path.join("datasets", dataset, "test.json")
        with open(eval_path) as fd:
            instances = json.load(fd)

        num_correct = 0
        num_samples = 0
        miss = 0.001
        for k in trange(0, len(instances), args.batch_size, disable=not dist.is_main(), desc=dataset):
            start_time = time.time()
            batch = instances[k : k + args.batch_size]
            targets = [instance["answer"] for instance in batch]

            inputs = [generate_prompt(instance["instruction"]) for instance in batch]
            input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True).input_ids.cuda()
            
            with torch.inference_mode():
                output_ids = model.generate(input_ids, generation_config=generation_config)
            
            if k > 32:  # warmup
                times[dataset].append(time.time() - start_time)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [output.split("### Response:")[1].strip() for output in outputs]
            outputs_ext = [extract_answer(output, dataset=dataset) for output in outputs]
            
            for input, output, output_ext, target in zip(inputs, outputs, outputs_ext, targets):
                if dataset in ["gsm8k", "mawps", "svamp"]:
                    correct = int(abs(float(target) - output_ext) <= miss)
                else:
                    correct = int(output_ext == target)
                num_correct += correct
                results[dataset].append({
                    "input": input,
                    "output": output,
                    "output_ext": output_ext,
                    "target": target,
                    "correct": correct
                })
            
            num_samples += len(targets)

        metrics[dataset] = num_correct / num_samples

        print(f"Average inference time for {dataset}: {sum(times[dataset]) / len(times[dataset]):.4f} seconds") 



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--spft", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--enable_static", type=bool, default=False)
    parser.add_argument("--act_channel", type=str, default=None)

    # parser.add_argument("--base_path", type=str)
    #* WeLore args
    # parser.add_argument("--we_lore", type=int, default=0)
    # parser.add_argument("--we_lore_path_rank_k_checkpoint", type=str)
    # parser.add_argument("--we_lore_singular_value_path", type=str)
    # parser.add_argument("--we_lore_model_rank", type=int, default=50)
    # parser.add_argument("--we_lore_min_ratio", type=float, default=0.4999)
    
    args = parser.parse_args()

    main(args)
