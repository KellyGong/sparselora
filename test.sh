CUDA_VISIBLE_DEVICES=0 python spft/test/main.py \
    --model_name_or_path checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/math/lora/math_10k_b32_ep3.0_lr0.0003_peft-lora_svd_8_ffn=0.99x28-qkv=0.75x14_skip-out/43 \
    --dataset gsm8k+svamp+mawps

CUDA_VISIBLE_DEVICES=0 python spft/test/main.py \
    --dataset gsm8k+svamp+mawps \
    --model_name_or_path checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/math/include_linear_lora/lora/math_10k_b16_ep3.0_lr0.0003_peft-lora_svd_8_ffn=0.99x28-qkv=0.75x14_skip-out/41 \
    --spft configs/sparsity/llama3-8b-math10k.yaml \
    --enable_static True \
    --act_channel configs/act_freq/math_10k.json