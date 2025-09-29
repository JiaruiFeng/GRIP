import random
from typing import Optional

import numpy as np
from transformers import PreTrainedTokenizer

from constants import ANSWER_TAG, SYSTEM_PROMPT
from grip.tasks.utils import shuffle_graph, clean_graph, compute_node_edge_weight, weighted_sampling
from models import BaseInferenceModel
from .gen_base import GenGraphTaskBase


class GenContextQATask(GenGraphTaskBase):
    gen_system_prompt = SYSTEM_PROMPT

    answer_format = f"<{ANSWER_TAG}>" + "{answer}" + f"</{ANSWER_TAG}>"

    template = """Given the context graph titled {title}, please answer the following question: {question} Response in the following format:""" + answer_format.format(
        answer="[answer]")

    node_attribute_user_prompt = """
    You are given one sentence describe the information of an entity. Your task is to randomly mask one attribute of the entity and generate a qeustion
     and answer to ask about the masked part.  Given one sentence, there are multiple way to generate the QA pair. For example, given sentence 
     "A green apple is on the table at position (5, 10)",

     Q1: What is the color of the apple on the table at position (5, 10)?
     A1: green.

     Q2: Where is the green apple at position (5, 10)?
     A2: on the table.

     Q3: What is the position of the green apple on the table?
     A3: (5, 10).

     Q4L which entity is green and on the table at position (5, 10)?
     A4: apple.

    You are encourage to mask the semantic attribute or entity itself over numeric attribute. 

    Here is the sentence:
    {node}

    Please provide your answer in the following format: \nQuestion: [question]. \n\nAnswer: [answer].
    Please DON'T output quotes and strictly follow the format.

    """

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

        rephrase_tasks = []
        graph_index = []
        node_attribute_tasks = []
        for index, (graph, title) in enumerate(zip(self.graph_list, self.title_list)):
            node_list = graph["node_list"]
            edge_list = graph["edge_list"]
            edge_index = np.array(graph["edge_index"])

            # clean graph, remove repeated edges and self-loops
            edge_list, edge_index = clean_graph(edge_list, edge_index)

            # random shuffle the graph order
            node_list, edge_list, edge_index = shuffle_graph(node_list, edge_list, edge_index)

            num_nodes = len(node_list)
            node_weight, edge_weight = compute_node_edge_weight(edge_index, num_nodes)

            for _ in range(self.num_qa):
                sampled_node = weighted_sampling(list(range(len(node_list))), node_weight, 1)[0]
                node = node_list[sampled_node]
                node_attribute_tasks.append(self.node_attribute_user_prompt.format(node=node))

                sampled_edge = weighted_sampling(list(range(len(edge_list))), edge_weight, 1)[0]
                edge = edge_list[sampled_edge]
                question_type = random.choice(["lp", "src", "tgt"])
                if question_type == "lp":
                    question = "what is the relation between {src} and {tgt}?"
                    question = question.format(src=edge[0], tgt=edge[2])
                    rephrase_tasks.append(self.rephrase_user_prompt.format(question=question, answer=edge[1]))
                elif question_type == "src":
                    question = "which entity has the relation {rel} to {tgt}?"
                    question = question.format(rel=edge[1], tgt=edge[2])
                    rephrase_tasks.append(self.rephrase_user_prompt.format(question=question, answer=edge[0]))
                else:
                    question = "Which entity has the relation {rel} from {src}?"
                    question = question.format(src=edge[0], rel=edge[1])
                    rephrase_tasks.append(self.rephrase_user_prompt.format(question=question, answer=edge[2]))
                graph_index.append(index)

        qa_tasks = [[] for _ in range(len(self.graph_list))]
        # rephrase
        rephrase_results = self.task_generator.inference(rephrase_tasks, self.gen_system_prompt)
        for r_text, g_index in zip(rephrase_results, graph_index):
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

        # node attribute
        node_attribute_results = self.task_generator.inference(node_attribute_tasks, self.gen_system_prompt)
        for n_text, g_index in zip(node_attribute_results, graph_index):
            n_text = n_text["response"]
            n_texts = n_text.strip().split("\n\n")
            if len(n_texts) != 2:
                continue
            question, answer = n_texts
            question = question.split(":")[-1].strip()
            answer = answer.split(":")[-1].strip()
            question = self.template.format(title=self.title_list[g_index], question=question)
            answer = self.answer_format.format(answer=answer)
            sample = self.create_chat_message(question, answer)
            qa_tasks[g_index].append(sample)
        
        return self.sample_post_process(qa_tasks)

