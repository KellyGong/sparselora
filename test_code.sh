CUDA_VISIBLE_DEVICES=0 python spft/test/code_test.py --model_name_or_path NousResearch/Meta-Llama-3-8B-Instruct

evalplus.sanitize --samples generated_completions.jsonl
evalplus.syncheck --samples generated_completions-sanitized.jsonl --dataset humaneval
evalplus.evaluate --dataset humaneval --samples generated_completions-sanitized.jsonl