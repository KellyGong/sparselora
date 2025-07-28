CUDA_VISIBLE_DEVICES=1 python spft/train/main.py \
    --report_to none \
    --logging_strategy no \
    --output_dir checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/csr170k \
    --seed 43 \
    --model_name_or_path NousResearch/Meta-Llama-3-8B-Instruct \
    --spft configs/sparsity/llama3-8b-csr.yaml \
    --benchmark True \
    --spft_start_step 0 \
    --config configs/train/csr170k_train.yaml \
    --do_eval True

CUDA_VISIBLE_DEVICES=0 python spft/train/main.py \
    --report_to none \
    --logging_strategy no \
    --output_dir checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/math \
    --seed 41 \
    --model_name_or_path NousResearch/Meta-Llama-3-8B-Instruct \
    --spft configs/sparsity/llama3-8b-math10k.yaml \
    --benchmark True \
    --spft_start_step 0 \
    --config configs/train/math10k_train.yaml \
    --do_eval True

CUDA_VISIBLE_DEVICES=1 python spft/train/main.py \
    --report_to none \
    --logging_strategy no \
    --output_dir checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/math \
    --seed 41 \
    --model_name_or_path NousResearch/Meta-Llama-3-8B-Instruct \
    --spft configs/sparsity/llama3-8b-math10k.yaml \
    --benchmark True \
    --spft_start_step 0 \
    --config configs/train/math10k_train.yaml \
    --enable_static True \
    --act_channel configs/act_freq/math_10k.json \
    --do_eval True
