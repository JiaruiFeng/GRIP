import gc

import torch
from transformers import PreTrainedTokenizer
from vllm.distributed.parallel_state import destroy_model_parallel

from models import get_inf_model
from .gen_base import GenGraphTaskBase
from .gen_context import GenGraphContextTask
from .gen_context_qa import GenContextQATask
from .gen_reasoning_qa import GenReasoningQATask
from .gen_summarization import GenSummarizationTask
from .task_dataset import TaskDataset

CONTEXT_TASK_DICT = {
    "context": GenGraphContextTask,
    "summarization": GenSummarizationTask,
}

QA_TASK_DICT = {
    "context_qa": GenContextQATask,
    "reasoning_qa": GenReasoningQATask,
}

LLM_GEN_TASKS = ("context_qa", "summarization", "reasoning_qa")


class GripTaskGeneration():
    gen_tasks = ["context", "summarization", "context_qa", "reasoning_qa"]

    def __init__(
            self,
            graph_list: list[list[list[str]]],
            title_list: list[str],
            tokenizer: PreTrainedTokenizer,
            refer_data: list[dict] = None,
            task_generator_model_name: str = "qwen-32b",
            task_gen_max_length: int = 1000,
            involve_qa_epochs: int = 0,
            **kwargs,
    ):
        self.graph_list = graph_list
        self.title_list = title_list
        self.tokenizer = tokenizer
        self.refer_data = refer_data
        self.task_generator_model_name = task_generator_model_name
        self.task_gen_max_length = task_gen_max_length
        self.involve_qa_epochs = involve_qa_epochs
        self.kwargs = kwargs
        self.task_generator = None

    def gen_task(self) -> list:
        if self.task_generator is None and any(t in LLM_GEN_TASKS for t in self.gen_tasks):
            self.load_model()
        context_task = [[] for _ in range(len(self.graph_list))]
        for task_name in CONTEXT_TASK_DICT.keys():
            task = CONTEXT_TASK_DICT[task_name](
                graph_list=self.graph_list,
                title_list=self.title_list,
                tokenizer=self.tokenizer,
                task_generator=self.task_generator,
                task_generator_model_name=self.task_generator_model_name,
                task_gen_max_length=self.task_gen_max_length,
                refer_data=self.refer_data,
                **self.kwargs
            )
            if task_name in self.gen_tasks:
                task_result = task()
            else:
                task_result = task(gen_empty_task=True)
            for i in range(len(self.graph_list)):
                context_task[i].extend(task_result[i])

        qa_task = [[] for _ in range(len(self.graph_list))]
        if self.involve_qa_epochs > 0:
            for task_name in QA_TASK_DICT.keys():
                task = QA_TASK_DICT[task_name](
                    graph_list=self.graph_list,
                    title_list=self.title_list,
                    tokenizer=self.tokenizer,
                    task_generator=self.task_generator,
                    task_generator_model_name=self.task_generator_model_name,
                    task_gen_max_length=self.task_gen_max_length,
                    refer_data=self.refer_data,
                    **self.kwargs
                )
                if task_name in self.gen_tasks:
                    task_result = task()
                else:
                    task_result = task(gen_empty_task=True)
                for i in range(len(self.graph_list)):
                    qa_task[i].extend(task_result[i])
        self.unload_model()
        return [TaskDataset(context_samples=c, qa_samples=qa, tokenizer=self.tokenizer) for c, qa in
                zip(context_task, qa_task)]

    def __call__(self) -> list:
        return self.gen_task()

    def load_model(self):
        self.task_generator = get_inf_model(
            model_name=self.task_generator_model_name,
            use_vllm=True,
            gen_max_length=self.task_gen_max_length,
            tokenize_max_length=10000,
            top_p=0.9,
            temperature=0.6,
        )

    def unload_model(self):
        if self.task_generator is not None:
            # torch.distributed.destroy_process_group()
            destroy_model_parallel()
            # del self.task_generator.model.llm_engine.model_executor.driver_worker
            del self.task_generator.model
            del self.task_generator
            self.task_generator = None
            torch.cuda.empty_cache()
            gc.collect()
