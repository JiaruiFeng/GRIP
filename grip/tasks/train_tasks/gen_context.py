from typing import Optional

import numpy as np
from transformers import PreTrainedTokenizer

from constants import ANSWER_TAG, SYSTEM_PROMPT
from grip.tasks.utils import shuffle_graph, clean_graph
from models import BaseInferenceModel
from .gen_base import GenGraphTaskBase


class GenGraphContextTask(GenGraphTaskBase):
    gen_system_prompt = SYSTEM_PROMPT

    answer_format = f"<{ANSWER_TAG}>" + "{answer}" + f"</{ANSWER_TAG}>"

    node_template = """Given the context graph titled {title}, recite one node in the graph. """

    node_context_format = "\n{node}."

    edge_template = """Given the context graph titled {title}, recite one edge in the graph. """

    edge_context_format = "\nthe node {src} is {rel} the node {tgt}."

    rephrase_user_prompt = """
    You are given one factual sentence. Rewrite it to preserve the original meaning and all factual/numeric details, but 
    change the wording and structure (e.g., reorder clauses, switch active↔passive, use synonyms). Do not add or omit 
    information.
    The sentence is provided below:
    {context}
    
    Please answer in the following format: \nRephrased: [rephrased sentence].
    Please DON'T output quotes and strictly follow the format.
    """

    def __init__(
            self,
            graph_list: list,
            title_list: list[str],
            tokenizer: PreTrainedTokenizer,
            task_generator: Optional[BaseInferenceModel] = None,
            task_generator_model_name: str = "qwen-32b",
            task_gen_max_length: int = 1000,
            **kwargs,
    ):
        super().__init__(
            graph_list=graph_list,
            title_list=title_list,
            tokenizer=tokenizer,
            task_generator=task_generator,
            task_generator_model_name=task_generator_model_name,
            task_gen_max_length=task_gen_max_length,
            **kwargs,
        )

    def gen_task(self, gen_empty_task=False) -> list:
        if gen_empty_task:
            return [[] for _ in range(len(self.graph_list))]

        train_context_result = []
        rephrase_tasks = []
        rephrase_graph_index = []
        content_type = []
        for index, (graph, title) in enumerate(zip(self.graph_list, self.title_list)):
            sample_train_context = []
            node_list = graph["node_list"]
            edge_list = graph["edge_list"]
            edge_index = np.array(graph["edge_index"])

            # clean graph, remove repeated edges and self-loops
            edge_list, edge_index = clean_graph(edge_list, edge_index)

            # random shuffle the graph order
            node_list, edge_list, edge_index = shuffle_graph(node_list, edge_list, edge_index)
            for node in node_list:
                node_text = self.node_context_format.format(node=node, title=title)
                rephrase_tasks.append(self.rephrase_user_prompt.format(context=node_text))
                rephrase_graph_index.append(index)
                content_type.append("node")
                answer = self.answer_format.format(answer=node_text)
                sample = self.create_chat_message(self.node_template.format(title=title), answer)
                sample_train_context.append(sample)

            # plain edge information
            for edge in edge_list:
                edge_text = self.edge_context_format.format(src=edge[0], tgt=edge[2], rel=edge[1], title=title)
                rephrase_tasks.append(self.rephrase_user_prompt.format(context=edge_text))
                rephrase_graph_index.append(index)
                content_type.append("edge")
                answer = self.answer_format.format(answer=edge_text)
                sample = self.create_chat_message(self.edge_template.format(title=title), answer)
                sample_train_context.append(sample)

            train_context_result.append(sample_train_context)
        # rephrase
        rephrase_result = self.task_generator.inference(rephrase_tasks, self.gen_system_prompt)
        for r_text, g_index, type in zip(rephrase_result, rephrase_graph_index, content_type):
            r_text = r_text["response"]
            r_text = r_text.strip()
            if r_text.startswith("Rephrased:"):
                r_text = r_text[len("Rephrased:"):].strip()
            answer = self.answer_format.format(answer=r_text)
            if type == "node":
                sample = self.create_chat_message(self.node_template.format(title=self.title_list[g_index]), answer)
            else:
                sample = self.create_chat_message(self.edge_template.format(title=self.title_list[g_index]), answer)
            train_context_result[g_index].append(sample)

        return train_context_result
