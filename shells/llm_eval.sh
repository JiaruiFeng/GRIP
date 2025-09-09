PYTHONPATH=. python scripts/run_eval.py \
    --dataset_name scene_graph \
    --input_file outputs/llm_inf/scene_graph/llm_inf.json \
    --output_file outputs/llm_inf/scene_graph/llm_eval.json \
    --llm_as_judge_model qwen-32b \
    --tokenize_max_length 30000 \
    --gen_max_length 1000 \


