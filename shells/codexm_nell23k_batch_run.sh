#PYTHONPATH=. python scripts/mp_wrapper.py \
#    --script scripts/run_llm.py \
#    --num_process 1 \
#    --dataset codexm \
#    --output_file codexm_qwen_baseline_submit_nocontext.json \
#    --do_eval \
#    --metrics em f1 hit \
#    --llm_as_judge_model qwen-32b \
#    --report_to_wandb \
#    --wandb_project_name grip \
#    --wandb_run_name codexm_qwen_baseline_submit_nocontext \
#    --subprocess_args \
#    --overwrite True \
#    --tokenize_max_length 30000 \
#    --gen_max_length 1000 \
#    --padding_side left \
#    --truncation_side left \
#    --model_name qwen-7b \
#    --dtype bfloat16 \
#    --use_vllm True \
#    --batch_size -1 \
#    --do_sample True \
#    --top_p 0.9 \
#    --temperature 0.6 \
#    --no_graph_context True \
#    --use_subgraph True \
#    --index_format False \
#    --report_input_token_count True > logs/qwen_llm_inf_codexm_nocontext.log 2>&1
#
#PYTHONPATH=. python scripts/mp_wrapper.py \
#    --script scripts/run_llm.py \
#    --num_process 1 \
#    --dataset nell23k \
#    --output_file nell23k_qwen_baseline_submit_nocontext.json \
#    --do_eval \
#    --metrics em f1 hit \
#    --llm_as_judge_model qwen-32b \
#    --report_to_wandb \
#    --wandb_project_name grip \
#    --wandb_run_name nell23k_qwen_baseline_submit_nocontext \
#    --subprocess_args \
#    --overwrite True \
#    --tokenize_max_length 30000 \
#    --gen_max_length 1000 \
#    --padding_side left \
#    --truncation_side left \
#    --model_name qwen-7b \
#    --dtype bfloat16 \
#    --use_vllm True \
#    --batch_size -1 \
#    --do_sample True \
#    --top_p 0.9 \
#    --temperature 0.6 \
#    --no_graph_context True \
#    --use_subgraph True \
#    --index_format False \
#    --report_input_token_count True > logs/qwen_llm_inf_nell23k_nocontext.log 2>&1
#
PYTHONPATH=. python scripts/mp_wrapper.py \
    --script scripts/run_llm.py \
    --num_process 1 \
    --dataset codexm \
    --output_file codexm_llama_baseline_context_submit_rerun.json \
    --do_eval \
    --metrics em f1 hit \
    --llm_as_judge_model qwen-32b \
    --report_to_wandb \
    --wandb_project_name grip \
    --wandb_run_name codexm_llama_baseline_context_submit_rerun \
    --subprocess_args \
    --overwrite True \
    --tokenize_max_length 30000 \
    --gen_max_length 1000 \
    --padding_side left \
    --truncation_side left \
    --model_name llama3-8b \
    --dtype bfloat16 \
    --use_vllm True \
    --do_sample True \
    --top_p 0.9 \
    --temperature 0.6 \
    --no_graph_context False \
    --use_subgraph True \
    --index_format False \
    --report_input_token_count True > logs/llama3_llm_inf_codexm_rerun.log 2>&1

PYTHONPATH=. python scripts/mp_wrapper.py \
    --script scripts/run_llm.py \
    --num_process 1 \
    --dataset nell23k \
    --output_file nell23k_llama_baseline_context_submit_rerun.json \
    --do_eval \
    --metrics em f1 hit \
    --llm_as_judge_model qwen-32b \
    --report_to_wandb \
    --wandb_project_name grip \
    --wandb_run_name nell23k_llama_baseline_context_submit_rerun \
    --subprocess_args \
    --overwrite True \
    --tokenize_max_length 30000 \
    --gen_max_length 1000 \
    --padding_side left \
    --truncation_side left \
    --model_name llama3-8b \
    --dtype bfloat16 \
    --use_vllm True \
    --do_sample True \
    --top_p 0.9 \
    --temperature 0.6 \
    --no_graph_context False \
    --use_subgraph True \
    --index_format False \
    --report_input_token_count True > logs/llama3_llm_inf_nell23k_rerun.log 2>&1

PYTHONPATH=. python scripts/mp_wrapper.py \
    --script scripts/run_llm.py \
    --num_process 1 \
    --dataset codexm \
    --output_file codexm_llama_baseline_context_submit_nocontext_rerun.json \
    --do_eval \
    --metrics em f1 hit \
    --llm_as_judge_model qwen-32b \
    --report_to_wandb \
    --wandb_project_name grip \
    --wandb_run_name codexm_llama_baseline_context_submit_nocontext_rerun \
    --subprocess_args \
    --overwrite True \
    --tokenize_max_length 30000 \
    --gen_max_length 1000 \
    --padding_side left \
    --truncation_side left \
    --model_name llama3-8b \
    --dtype bfloat16 \
    --use_vllm True \
    --do_sample True \
    --top_p 0.9 \
    --temperature 0.6 \
    --no_graph_context True \
    --use_subgraph True \
    --index_format False \
    --report_input_token_count True > logs/llama3_llm_inf_codexm_nocontext_rerun.log 2>&1

PYTHONPATH=. python scripts/mp_wrapper.py \
    --script scripts/run_llm.py \
    --num_process 1 \
    --dataset nell23k \
    --output_file nell23k_llama_baseline_context_submit.json_nocontext_rerun \
    --do_eval \
    --metrics em f1 hit \
    --llm_as_judge_model qwen-32b \
    --report_to_wandb \
    --wandb_project_name grip \
    --wandb_run_name nell23k_llama_baseline_context_submit_nocontext_rerun \
    --subprocess_args \
    --overwrite True \
    --tokenize_max_length 30000 \
    --gen_max_length 1000 \
    --padding_side left \
    --truncation_side left \
    --model_name llama3-8b \
    --dtype bfloat16 \
    --use_vllm True \
    --do_sample True \
    --top_p 0.9 \
    --temperature 0.6 \
    --no_graph_context True \
    --use_subgraph True \
    --index_format False \
    --report_input_token_count True > logs/llama3_llm_inf_nell23k_nocontext_rerun.log 2>&1
