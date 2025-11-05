import argparse
import json
import os
import re

import torch
from peft import AutoPeftModelForCausalLM
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from tabulate import tabulate
from tqdm import trange
from collections import defaultdict

from spft.utils import distributed as dist
from spft.api import SPFTConfig, load_channel_act_file, get_spft_model
from transformers import AutoTokenizer, GenerationConfig


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

    llama_usage = True if "Qwen" not in args.model_name_or_path else False

    model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    # device_map="auto",
    # max_memory=max_memory,
    ).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=512,
        padding_side="left",
        use_fast=False,
    )

    if len(tokenizer) > 32000 and llama_usage: #* Llama3
        print("Using LLaMA 3 tokenizer")
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
    
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    metrics = {}
    results = defaultdict(list)
    for dataset in args.dataset.split("+") if "+" in args.dataset else [args.dataset]:
        with open(os.path.join("datasets", dataset, "test.json")) as fd:
            instances = json.load(fd)
        instances = instances[dist.rank() :: dist.size()]

        num_correct = 0
        num_samples = 0
        miss = 0.001
        for k in trange(0, len(instances), args.batch_size, disable=not dist.is_main(), desc=dataset):
            batch = instances[k : k + args.batch_size]
            targets = [instance["answer"] for instance in batch]

            inputs = [generate_prompt(instance["instruction"]) for instance in batch]
            input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True).input_ids.cuda()
            
            with torch.inference_mode():
                output_ids = model.generate(input_ids, generation_config=generation_config)

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

    print(tabulate(metrics.items(), headers=["Dataset", "Accuracy"], tablefmt="simple_outline"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=32)

    args = parser.parse_args()

    main(args)
