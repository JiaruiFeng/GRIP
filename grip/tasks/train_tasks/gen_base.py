import gc
import random
from abc import abstractmethod, ABC

import torch
from transformers import PreTrainedTokenizer
from vllm.distributed.parallel_state import destroy_model_parallel

from constants import SYSTEM_PROMPT
from models import get_inf_model
from .train_system_prompts import TRAIN_SYSTEM_PROMPTS


class GenGraphTaskBase(ABC):
    task_system_prompts = TRAIN_SYSTEM_PROMPTS + [SYSTEM_PROMPT]

    def __init__(
            self,
            graph_list: list,
            title_list: list[str],
            tokenizer: PreTrainedTokenizer,
            task_generator_model_name: str = "qwen-32b",
            task_gen_max_length: int = 1000,
            **kwargs,
    ):
        self.graph_list = graph_list
        self.title_list = title_list
        self.tokenizer = tokenizer
        self.task_generator_model_name = task_generator_model_name
        self.task_gen_max_length = task_gen_max_length
        self.task_generator = None
        self.kwargs = kwargs
        super().__init__()

    @abstractmethod
    def gen_task(self) -> list:
        """
        generate train task.
        """
        pass

    def create_chat_message(self, question, answer):

        message = [
            {
                "role": "system",
                "content": random.sample(self.task_system_prompts, 1)[0]
                ,
            },
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            }
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False)

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

    def sample_post_process(self, text_list: list):
        end = self.tokenizer.eos_token + "\n"
        if text_list and isinstance(text_list[0], list):
            return [self.sample_post_process(text) for text in text_list]
        else:
            return_list = []
            for text in text_list:
                # if no end of token is appended, manually append it.
                if text[-len(end):] != end:
                    text += end
                return_list.append(text)
            return return_list
