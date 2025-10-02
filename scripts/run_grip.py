import os
from typing import Optional

import torch
from tqdm import tqdm, trange
from transformers import TrainingArguments

from arguments import (
    BaseArguments,
    TaskArguments,
    ModelArguments,
    CustomTrainingArguments,
    InferenceArguments,
    parse_args
)
from grip.tasks import GripTaskGeneration, gen_eval_task
from grip.training import train
from models import get_ft_model, load_peft_model
from utils import extract_tag_content, save_entry_to_list_json, load_list_json, set_random_seed, Timer


def extract_all_evidence_and_answers(text, add_evidence=True):
    if add_evidence:
        evidences = extract_tag_content(text, "evidence")
        evidences = "; ".join(evidences)
    else:
        evidences = ""

    answers = extract_tag_content(text, "answer")
    answers = "; ".join(answers)
    return evidences, answers


def run(
        rank: int,
        input_data: list[dict],
        exp_args: dict,
        training_args: TrainingArguments,
        output_file: str,
        ref_data: Optional[list[dict]] = None,
        num_resumed: int = 0,
):
    _ = set_random_seed()
    # get model and tokenizer
    model, tokenizer = get_ft_model(**exp_args)

    graph_list = [data["graph"] for i, data in enumerate(input_data) if i >= num_resumed]
    title_list = [data["title"] for i, data in enumerate(input_data) if i >= num_resumed]
    task_data = GripTaskGeneration(graph_list, title_list, tokenizer=tokenizer, refer_data=ref_data, **exp_args)()
    timer = Timer()
    # start training
    for i in tqdm(range(len(input_data)), desc=f"Rank {rank}, Sample: "):
        if i < num_resumed:
            continue
        input_d, task_d = input_data[i], task_data[i - num_resumed]
        model = load_peft_model(model, **exp_args)
        model, tokenizer = train(
            model=model,
            tokenizer=tokenizer,
            training_dataset=task_d,
            training_args=training_args,
            eval_dataset=None,
            **exp_args,
        )

        # inference
        if exp_args["continue_training"]:
            model = model.merge_and_unload()
        if model.device.type == "cpu":
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        eval_dataset = gen_eval_task(
            eval_mode="grip",
            input_data=input_d,
            tokenizer=tokenizer,
            **exp_args,
        )
        output_results = []
        output_nosample_results = []
        timer.start()
        for j in trange(0, len(eval_dataset), 1, desc=f"Inference.", disable=False, ):
            input_ids, Q, A = eval_dataset[j]
            if isinstance(A, str):
                A = [A]

            input_ids = input_ids.to(model.device)
            attention_mask = torch.ones_like(input_ids)
            # output = model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     pad_token_id=tokenizer.pad_token_id,
            #     eos_token_id=tokenizer.eos_token_id,
            #     max_new_tokens=exp_args["gen_max_length"],
            #     do_sample=exp_args["do_sample"],
            #     top_p=exp_args["top_p"],
            #     temperature=exp_args["temperature"],
            # )

            output_no_sample = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=exp_args["gen_max_length"],
                do_sample=False
            )
             
            output = output_no_sample


            response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            response_no_sample = tokenizer.decode(output_no_sample[0][input_ids.shape[-1]:], skip_special_tokens=True)
            evidence, answer = extract_all_evidence_and_answers(response, False)
            evidence, answer_no_sample = extract_all_evidence_and_answers(response_no_sample, False)

            output_result = {
                "id": input_d["id"],
                "question": Q,
                "raw_response": response.strip(),
                "response": answer.strip(),
                "raw_response_no_sample": response_no_sample.strip(),
                "response_no_sample": answer_no_sample.strip(),
                "evidence": evidence.strip(),
                "target": A}
            output_results.append(output_result)
        timer.end()
        # save
        save_entry_to_list_json(output_file, output_results)
    print(f"Total inference time: {timer.return_time()}")


def main():
    base_args, training_args, exp_args = parse_args(
        (
            BaseArguments,
            TrainingArguments,
            (TaskArguments, ModelArguments, CustomTrainingArguments, InferenceArguments)),
        no_dict=(TrainingArguments,)
    )
    # set up experiment
    rank = base_args.pop("rank")
    input_file = base_args.pop("input_file")
    ref_file = base_args.pop("ref_file")
    output_file = base_args.pop("output_file")
    overwrite = base_args.pop("overwrite")

    num_resumed = 0
    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            with open(output_file, 'r') as f:
                num_resumed = len(f.readlines())
    input_data = load_list_json(input_file)
    if num_resumed >= len(input_data):
        return

    if os.path.exists(ref_file):
        ref_data = load_list_json(ref_file)
    else:
        ref_data = None
    run(rank, input_data, exp_args, training_args, output_file, ref_data, num_resumed)


if __name__ == "__main__":
    main()
