CUDA_VISIBLE_DEVICES=0 python spft/test/main.py \
    --model_name_or_path checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/math_punc/reft/math_10k_b16_ep3.0_lr0.0003_peft-reft_dense/42 \
    --spft configs/sparsity/llama3-8b-dense.yaml \
    --dataset gsm8k+svamp+mawps

CUDA_VISIBLE_DEVICES=0 python spft/test/main.py \
    --dataset gsm8k+svamp+mawps \
    --model_name_or_path checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/math/include_linear_lora/lora/math_10k_b16_ep3.0_lr0.0003_peft-lora_svd_8_ffn=0.99x28-qkv=0.75x14_skip-out/41 \
    --spft configs/sparsity/llama3-8b-math10k.yaml \
    --enable_static True \
    --act_channel configs/act_freq/math_10k.json

CUDA_VISIBLE_DEVICES=1 python spft/test/main.py \
    --model_name_or_path checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/csr170k/reft/commonsense_170k_b16_ep1.0_lr0.0003_peft-reft_dense/41 \
    --spft configs/sparsity/llama3-8b-dense.yaml \
    --dataset social-iqa+hellaswag+winogrande+arc-easy+arc-challenge+openbookqa

CUDA_VISIBLE_DEVICES=1 python spft/test/main.py \
    --dataset gsm8k+svamp+mawps \
    --model_name_or_path checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/math/dora/math_10k_b16_ep3.0_r4_lr0.0003_peft-dora_dense/41

CUDA_VISIBLE_DEVICES=0 python spft/test/main.py \
    --dataset social-iqa+hellaswag+winogrande+arc-easy+arc-challenge+openbookqa \
    --model_name_or_path checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/csr170k/dora/commonsense_170k_b8_ep1.0_r4_lr0.0003_peft-dora_dense/42


CUDA_VISIBLE_DEVICES=0 python spft/test/ttft_test.py \
    --dataset hellaswag \
    --model_name_or_path NousResearch/Meta-Llama-3-8B-Instruct