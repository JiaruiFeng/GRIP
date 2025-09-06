from .base import BaseEvalDataset
from .standard_eval import StandardEvalDataset
from .grip_eval import GRIPEvalDataset
from transformers import PreTrainedTokenizer

def gen_eval_task(
        eval_mode: str,
        input_data: dict,
        tokenizer: PreTrainedTokenizer,
    **kwargs,
) -> BaseEvalDataset:
    questions = input_data["questions"]
    answers = input_data["answers"]
    graph = input_data["graph"]
    title = input_data["title"]

    if eval_mode == "grip":
        return GRIPEvalDataset(
            questions=questions,
            answers=answers,
            graph=graph,
            title=title,
            tokenizer=tokenizer,
            **kwargs,
        )
    elif eval_mode == "standard":
        return StandardEvalDataset(
            questions=questions,
            answers=answers,
            graph=graph,
            title=title,
            tokenizer=tokenizer,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")
