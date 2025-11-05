CUDA_VISIBLE_DEVICES=0 python spft/train/main.py \
    --report_to none \
    --logging_strategy no \
    --output_dir checkpoints/Qwen/Qwen3-8B/math \
    --seed 41 \
    --model_name_or_path Qwen/Qwen3-8B \
    --spft configs/sparsity/qwen3-8b-math10k.yaml \
    --benchmark True \
    --spft_start_step 0 \
    --config configs/train/math10k_train.yaml \
    --do_eval True

CUDA_VISIBLE_DEVICES=0 python spft/train/main.py \
    --report_to none \
    --logging_strategy no \
    --output_dir checkpoints/Qwen/Qwen3-8B/math \
    --seed 41 \
    --model_name_or_path Qwen/Qwen3-8B \
    --spft configs/sparsity/qwen3-8b-dense.yaml \
    --benchmark True \
    --spft_start_step 0 \
    --config configs/train/math10k_train.yaml \
    --do_eval True