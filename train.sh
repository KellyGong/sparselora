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

CUDA_VISIBLE_DEVICES=1 python spft/train/main.py \
    --report_to none \
    --logging_strategy no \
    --output_dir checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/code \
    --seed 41 \
    --model_name_or_path NousResearch/Meta-Llama-3-8B-Instruct \
    --spft configs/sparsity/llama3-8b-dense.yaml \
    --benchmark True \
    --spft_start_step 0 \
    --config configs/train/codefeedback_train.yaml \
    --do_eval True


CUDA_VISIBLE_DEVICES=0 python spft/train/main.py \
    --report_to none \
    --logging_strategy no \
    --output_dir checkpoints/NousResearch/Meta-Llama-3-8B-Instruct/wizardlm \
    --seed 41 \
    --model_name_or_path NousResearch/Meta-Llama-3-8B-Instruct \
    --spft configs/sparsity/llama3-8b-dense.yaml \
    --benchmark True \
    --spft_start_step 0 \
    --config configs/train/wizardlm_train.yaml \
    --do_eval True


CUDA_VISIBLE_DEVICES=0 python spft/train/main.py \
    --report_to none \
    --logging_strategy no \
    --output_dir checkpoints/meta-llama/Llama-3.1-8B-Instruct/math \
    --seed 41 \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --spft configs/sparsity/llama3-8b-dense.yaml \
    --benchmark True \
    --spft_start_step 0 \
    --config configs/train/math10k_train.yaml \
    --do_eval True \
    --enable_static True \
    --act_channel configs/act_freq/math_10k.json
    
