CUDA_VISIBLE_DEVICES=0 python spft/test/main.py \
    --model_name_or_path checkpoints/Qwen/Qwen3-8B/math/lora/math_10k_b16_ep3.0_lr0.0003_peft-lora_svd_8_ffn=0.99x28-qkv=0.75x14_skip-out/41 \
    --dataset gsm8k+svamp+mawps

CUDA_VISIBLE_DEVICES=0 python spft/test/main.py \
    --model_name_or_path checkpoints/Qwen/Qwen3-8B/math/lora/math_10k_b16_ep3.0_lr0.0003_peft-lora_svd_8_ffn=0x28-qkv=0x14_skip-out/41/ \
    --dataset gsm8k+svamp+mawps