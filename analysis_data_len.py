import os
import json
from transformers import AutoTokenizer

# 使用 Llama-3 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

def analyze_folder(file_path):
    lengths = 0
    n = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 支持单条或多条数据
        for data_i in data:
            instr = data_i['instruction']
            tokens = tokenizer(instr)
            length = len(tokens['input_ids'])
            lengths += length
            n += 1
    
    print(lengths / n)

if __name__ == "__main__":
    folder = "datasets/openbookqa/test.json"  # 修改为你的文件夹路径
    analyze_folder(folder)