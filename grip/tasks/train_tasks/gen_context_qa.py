import random

import numpy as np
from transformers import PreTrainedTokenizer

from constants import ANSWER_TAG
from grip.tasks.utils import shuffle_graph, clean_graph
from .gen_base import GenGraphTaskBase


class GenContextQATask(GenGraphTaskBase):
    answer_format = f"<{ANSWER_TAG}>" + "{answer}" + f"</{ANSWER_TAG}>"

    template = """Given the context graph titled {title}, please answer the following question: {question} Response in the following format:""" + answer_format.format(
        answer="[answer]")

    def __init__(
            self,
            graph_list: list,
            title_list: list[str],
            tokenizer: PreTrainedTokenizer,
            task_generator_model_name: str = "qwen-32b",
            task_gen_max_length: int = 1000,
            **kwargs,
    ):
        super().__init__(
            graph_list=graph_list,
            title_list=title_list,
            tokenizer=tokenizer,
            task_generator_model_name=task_generator_model_name,
            task_gen_max_length=task_gen_max_length,
            **kwargs,
        )

    def gen_task(self, gen_empty_task=False) -> list:
        if gen_empty_task:
            return [[] for _ in range(len(self.graph_list))]

        qa_tasks = []
        for graph, title in zip(self.graph_list, self.title_list):
            sample_train_qa = []
            node_list = graph["node_list"]
            edge_list = graph["edge_list"]
            edge_index = np.array(graph["edge_index"])

            # clean graph, remove repeated edges and self-loops
            edge_list, edge_index = clean_graph(edge_list, edge_index)

            # random shuffle the graph order
            node_list, edge_list, edge_index = shuffle_graph(node_list, edge_list, edge_index)

            for edge in edge_list:
                question_type = random.choice(["lp", "src", "tgt"])
                if question_type == "lp":
                    question = "what is the relation between the node {src} and the node {tgt}?"
                    question = question.format(src=edge[0], tgt=edge[2])
                    question = self.template.format(title=title, question=question)
                    answer = self.answer_format.format(answer=edge[1])
                elif question_type == "src":
                    question = "which node has the relation {rel} to node {tgt}?"
                    question = question.format(rel=edge[1], tgt=edge[2])
                    question = self.template.format(title=title, question=question)
                    answer = self.answer_format.format(answer=edge[0])
                else:
                    question = "Which node has the relation {rel} from the node {src}?"
                    question = question.format(src=edge[0], rel=edge[1])
                    question = self.template.format(title=title, question=question)
                    answer = self.answer_format.format(answer=edge[2])

                train_task = self.create_chat_message(question, answer)
                sample_train_qa.append(train_task)
            qa_tasks.append(sample_train_qa)

        return self.sample_post_process(qa_tasks)
