PYTHONPATH=. python scripts/run_eval.py \
    --dataset_name scene_graph \
    --input_file outputs/grip_inf/scene_graph/scene_graph_qwen_latest_large.json \
    --output_file outputs/grip_inf/scene_graph/scene_graph_qwen_latest_large_wrong.json \
    --llm_as_judge_model qwen-32b \
    --tokenize_max_length 30000 \
    --gen_max_length 1000 \


