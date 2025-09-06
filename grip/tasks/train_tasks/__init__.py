from transformers import PreTrainedTokenizer

from constants import HF_DECODER_ONLY_LLMS
from .gen_base import GenGraphTaskBase
from .gen_context import GenGraphContextTask
from .gen_context_qa import GenContextQATask
from .gen_summarization import GenSummarizationTask
from .gen_reasoning_qa import GenReasoningQATask
from .task_dataset import TaskDataset


def gen_train_task(
        graph_list: list[list[list[str]]],
        title_list: list[str],
        tokenizer: PreTrainedTokenizer,
        ref_data: list[dict] = None,
        task_generator_model_name: str = "qwen-32b",
        task_gen_max_length: int = 1000,
        involve_qa_epochs: int = 0,
        **kwargs,
) -> list:
    raw_context_task = GenGraphContextTask(
        graph_list=graph_list,
        title_list=title_list,
        tokenizer=tokenizer,
        task_generator_model_name=task_generator_model_name,
        task_gen_max_length=task_gen_max_length,
        **kwargs
    ).gen_task()

    summarization_task = GenSummarizationTask(
        graph_list=graph_list,
        title_list=title_list,
        tokenizer=tokenizer,
        task_generator_model_name=task_generator_model_name,
        task_gen_max_length=task_gen_max_length,
        **kwargs
    ).gen_task()

    context_task = [c + s for c, s in zip(raw_context_task, summarization_task)]

    if involve_qa_epochs > 0:
        context_qa_task = GenContextQATask(
            graph_list=graph_list,
            title_list=title_list,
            tokenizer=tokenizer,
            task_generator_model_name=task_generator_model_name,
            task_gen_max_length=task_gen_max_length,
            **kwargs
        ).gen_task()
        reasoning_qa_task = GenReasoningQATask(
            graph_list=graph_list,
            title_list=title_list,
            tokenizer=tokenizer,
            refer_data=ref_data,
            task_generator_model_name=task_generator_model_name,
            task_gen_max_length=task_gen_max_length,
            **kwargs
        ).gen_task()
        qa_task = [c_qa + r_qa for c_qa, r_qa in zip(context_qa_task, reasoning_qa_task)]
    else:
        qa_task = [[] for _ in range(len(graph_list))]

    return [TaskDataset(context_samples=c, qa_samples=qa, tokenizer=tokenizer) for c, qa in
            zip(context_task, qa_task)]
