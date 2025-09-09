import random
from typing import Optional

import numpy as np
from transformers import PreTrainedTokenizer

from constants import ANSWER_TAG, SYSTEM_PROMPT
from grip.tasks.utils import shuffle_graph, clean_graph
from models import BaseInferenceModel
from .gen_base import GenGraphTaskBase


class GenContextQATask(GenGraphTaskBase):
    gen_system_prompt = SYSTEM_PROMPT

    answer_format = f"<{ANSWER_TAG}>" + "{answer}" + f"</{ANSWER_TAG}>"

    template = """Given the context graph titled {title}, please answer the following question: {question} Response in the following format:""" + answer_format.format(
        answer="[answer]")

    rephrase_user_prompt = """
    You are given one question and answer pair. Rewrite it to preserve the original meaning and all factual/numeric 
    details for both question and answer, but change the wording and structure (e.g., reorder clauses, switch 
    active↔passive, use synonyms). Do not add or omit information.
    The question and answer are provided below:
    Question: {question}
    Answer: {answer}

    Please answer in the following format: \nRephrased question: [rephrased sentence]. \n\nRephrased answer: [rephrased answer].
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
            num_context_qa: int = 10,
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
        self.num_qa = num_context_qa

    def gen_task(self, gen_empty_task=False) -> list:
        if gen_empty_task or self.num_qa <= 0:
            return [[] for _ in range(len(self.graph_list))]

        qa_tasks = []
        rephrase_tasks = []
        rephrase_graph_index = []
        for index, (graph, title) in enumerate(zip(self.graph_list, self.title_list)):
            sample_train_qa = []
            node_list = graph["node_list"]
            edge_list = graph["edge_list"]
            edge_index = np.array(graph["edge_index"])

            # clean graph, remove repeated edges and self-loops
            edge_list, edge_index = clean_graph(edge_list, edge_index)

            # random shuffle the graph order
            node_list, edge_list, edge_index = shuffle_graph(node_list, edge_list, edge_index)

            for _ in range(self.num_qa):
                edge = random.choice(edge_list)
                question_type = random.choice(["lp", "src", "tgt"])
                if question_type == "lp":
                    question = "what is the relation between the node {src} and the node {tgt}?"
                    question = question.format(src=edge[0], tgt=edge[2])
                    rephrase_tasks.append(self.rephrase_user_prompt.format(question=question, answer=edge[1]))
                    question = self.template.format(title=title, question=question)
                    answer = self.answer_format.format(answer=edge[1])
                elif question_type == "src":
                    question = "which node has the relation {rel} to node {tgt}?"
                    question = question.format(rel=edge[1], tgt=edge[2])
                    rephrase_tasks.append(self.rephrase_user_prompt.format(question=question, answer=edge[0]))
                    question = self.template.format(title=title, question=question)
                    answer = self.answer_format.format(answer=edge[0])
                else:
                    question = "Which node has the relation {rel} from the node {src}?"
                    question = question.format(src=edge[0], rel=edge[1])
                    rephrase_tasks.append(self.rephrase_user_prompt.format(question=question, answer=edge[2]))
                    question = self.template.format(title=title, question=question)
                    answer = self.answer_format.format(answer=edge[2])
                rephrase_graph_index.append(index)
                train_task = self.create_chat_message(question, answer)
                sample_train_qa.append(train_task)
            qa_tasks.append(sample_train_qa)

        # rephrase
        rephrase_results = self.task_generator.inference(rephrase_tasks, self.gen_system_prompt)
        for r_text, g_index in zip(rephrase_results, rephrase_graph_index):
            r_text = r_text["response"]
            r_texts = r_text.strip().split("\n\n")
            if len(r_texts) != 2:
                continue
            question, answer = r_texts
            question = question.split(":")[-1].strip()
            answer = answer.split(":")[-1].strip()
            question = self.template.format(title=self.title_list[g_index], question=question)
            answer = self.answer_format.format(answer=answer)
            sample = self.create_chat_message(question, answer)
            qa_tasks[g_index].append(sample)

        return self.sample_post_process(qa_tasks)
