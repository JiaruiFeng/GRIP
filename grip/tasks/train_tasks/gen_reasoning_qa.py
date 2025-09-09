import random
from typing import Optional

import numpy as np
from transformers import PreTrainedTokenizer

from constants import SYSTEM_PROMPT, ANSWER_TAG, TUPLE_DELIMITER
from grip.tasks.utils import sample_fixed_hop_size_neighbor, clean_graph, shuffle_graph
from models import BaseInferenceModel
from .gen_base import GenGraphTaskBase


class GenReasoningQATask(GenGraphTaskBase):
    gen_system_prompt = SYSTEM_PROMPT

    gen_multihop_user_prompt = """
    You are given a subgraph of real-world facts and sample questions from a similar graph. Create two new questions to 
    test understanding of the subgraph.
    1. Select two sample questions that best match the subgraph. They must differ in at least one dimension:
    Interrogative (who/what/how/is–are/when/where), Format (descriptive/comparative/reasoning/yes–no), Focus 
    (attributes/relationships/numerical details/multi-entity connections).
    
    2. For each selected sample, write a new question with the same interrogative, format, and focus, but the content 
    of question and answer is fully grounded in the provided subgraph. Specifically:
     - The new questions should require multi-hop/indirect reasoning and use ≥2 distinct facts from the subgraph.
     - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).
     - Use natural phrasing; no “graph”/“node(s)”.
     - The two new questions must target different facts and differ from each other.
     - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the subgraph.
     
    The subgraph and sample questions are provided below:
    --Subgraph--
    {context}
    --Sample Questions--
    {sample_questions}

    Please first provide the referred sample question and then provide question and answer in the following format: 
    \nReferred question: [referred question] \nQuestion: [question] \nAnswer: [answer] {tuple_delimiter}
    \nReferred question: [referred question] \nQuestion: [question] \nAnswer: [answer]
    Please DON'T output quotes and separate each questions by {tuple_delimiter}, strictly follow the format. 
    """

    gen_local_user_prompt = """
    You are given a subgraph of real-world facts and sample questions from a similar graph. Create two new questions to 
    test understanding of the subgraph.
    1. Select two sample questions that best match the subgraph. They must differ in at least one dimension:
    Interrogative (who/what/how/is–are/when/where), Format (descriptive/comparative/reasoning/yes–no), Focus 
    (attributes/relationships/numerical details/multi-entity connections).
    
    2. For each selected sample, write a new question with the same interrogative, format, and focus, but the content 
    of question and answer is fully grounded in the provided subgraph. Specifically:
     - The new questions should focus on retrieving a single fact using partial information (e.g., infer the entity 
     from an attribute like color, appearance, or infer an attribute from the entity).
     - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).
     - Use natural phrasing; no “graph”/“node(s)”.
     - The two new questions must target different facts and differ from each other.
     - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the subgraph.

    The subgraph and sample questions are provided below:
    --Subgraph--
    {context}
    --Sample Questions--
    {sample_questions}

    Please first provide the referred sample question and then provide question and answer in the following format: 
    \nReferred question: [referred question] \nQuestion: [question] \nAnswer: [answer] {tuple_delimiter}
    \nReferred question: [referred question] \nQuestion: [question] \nAnswer: [answer]
    Please DON'T output quotes and separate each questions by {tuple_delimiter}, strictly follow the format. 
    """

    gen_global_user_prompt = """
    You are given a subgraph of real-world facts and sample questions from a similar graph. Create two new questions to 
    test understanding of the subgraph.
    1. Select two sample questions that best match the subgraph. They must differ in at least one dimension:
    Interrogative (who/what/how/is–are/when/where), Format (descriptive/comparative/reasoning/yes–no), Focus 
    (attributes/relationships/numerical details/multi-entity connections).

    2. For each selected sample, write a new question with the same type and style, but the answer is fully grounded in 
    the subgraph. Specifically:
     - The new questions should focus on reasoning over the full subgraph and use >2 distinct facts from the subgraph.
     - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).
     - The answer must include >= 2 element and separate them with comma or "and".
     - Use natural phrasing; no “graph”/“node(s)”.
     - The two new questions must target different facts and differ from each other.
     - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the subgraph.

    The subgraph and sample questions are provided below:
    --Subgraph--
    {context}
    --Sample Questions--
    {sample_questions}

    Please first provide the referred sample question and then provide question and answer in the following format: 
    \nReferred question: [referred question] \nQuestion: [question] \nAnswer: [answer] {tuple_delimiter}
    \nReferred question: [referred question] \nQuestion: [question] \nAnswer: [answer]
    Please DON'T output quotes and separate each questions by {tuple_delimiter}, strictly follow the format. 
    """

    gen_s2_user_prompt = """
    You are given a subgraph representing real-world facts and an QA pair generated from it. Your task 
    is to evaluate the correctness of question and answer pair based on the subgraph. Here is the instruction:
    1. For each question and answer pair, determine if the question and answer is reasonable and fully relate to the subgraph. 
    2. If the question/answer is wrong, provide a CORRECTED question/answer based on the subgraph.
    3. If the answer or question is correct but can be further improved, provide an IMPROVED answer that is more concise and precise.
    4. If the answer is already correct and optimal, simply restate the original answer as the IMPROVED answer.
        
    The subgraph and QA pairs are provided below:
    --Subgraph--
    {context}
    --QA pair--
    Question : {question}
    Answer : {answer}
    Please first provide whether is answer is changed and then provide the final question and answer in the following format: 
    \nChanged: [yes or no] \nQuestion: [question] \nAnswer: [answer] 
    Please DON'T output quotes and strictly follow the format. 
    
    """

    answer_format = f"<{ANSWER_TAG}>" + "{answer}" + f"</{ANSWER_TAG}>"

    template = """Given the context graph titled {title}, please answer the following question: {question} Response in the following format:""" + answer_format.format(
        answer="[answer]")

    def __init__(
            self,
            graph_list: list,
            title_list: list[str],
            tokenizer: PreTrainedTokenizer,
            task_generator: Optional[BaseInferenceModel] = None,
            refer_data: Optional[list[dict]] = None,
            task_generator_model_name: str = "qwen-32b",
            task_gen_max_length: int = 1000,
            num_reason_qa: int = 10,
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
        self.num_qa = num_reason_qa

        train_question_list = []
        if refer_data is not None:
            for d in refer_data:
                questions = d["questions"]
                answers = d["answers"]
                for q, a in zip(questions, answers):
                    qa_pair = "Question: {question}\nAnswer: {answer}".format(question=q.strip(), answer=a.strip())
                    train_question_list.append(qa_pair)
        else:
            train_question_list = [
                "What are the main contributions of Ada Lovelace to the early history of computing?",
                "Which differences distinguish the philosophies of Confucianism and Daoism?",
                "How did the invention of the printing press influence literacy rates in Europe?",
                "How many plays are attributed to William Shakespeare?",
                "Who were the key allies of the United States during World War II?",
                "When was the Great Wall of China first constructed, and during which dynasty?",
                "In what ways did the Industrial Revolution reshape urban life in Britain?",
                "What is the relationship between photosynthesis, carbon dioxide, and the global climate system?",
                "Could the Apollo 13 mission have succeeded without the improvised solutions developed by NASA engineers?",
            ]
        self.train_question_list = train_question_list

    def gen_task(self, gen_empty_task=False) -> list:
        if gen_empty_task or self.num_qa <= 0:
            return [[] for _ in range(len(self.graph_list))]
        qa_s1_tasks = []
        qa_graph_index = []
        qa_title_list = []
        context_list = []
        for index, (graph, title) in enumerate(zip(self.graph_list, self.title_list)):
            edge_list = graph["edge_list"]
            node_list = graph["node_list"]
            edge_index = np.array(graph["edge_index"])

            # clean graph, remove repeated edges and self-loops
            edge_list, edge_index = clean_graph(edge_list, edge_index)

            # random shuffle the graph order
            node_list, edge_list, edge_index = shuffle_graph(node_list, edge_list, edge_index)
            num_nodes = len(node_list)

            question_type = 0
            for _ in range(self.num_qa):
                root_node = int(np.random.choice(np.arange(len(node_list))))
                subgraph_nodes, subgraph_edges = sample_fixed_hop_size_neighbor(
                    edge_index.T,
                    np.arange(len(edge_list)),
                    root_node,
                    num_nodes,
                    hop=3,
                    max_nodes_per_hop=3,
                )
                subgraph_edges = [edge_list[e] for e in subgraph_edges.data]

                context = "The root node is {root}. The subgraph including edges: ".format(root=node_list[root_node])
                entry_format = "the source node {src} is {rel} the target node {tgt}; "
                for entry in subgraph_edges:
                    context += entry_format.format(src=entry[0], rel=entry[1], tgt=entry[2])

                k_shot_examples = random.sample(self.train_question_list, k=8)
                k_shot_examples = "; ".join(k_shot_examples)
                gen_prompt = [self.gen_local_user_prompt,
                              self.gen_global_user_prompt,
                              self.gen_multihop_user_prompt][question_type]

                qa_task = gen_prompt.format(context=context, tuple_delimiter=TUPLE_DELIMITER,
                                            sample_questions=k_shot_examples)
                qa_s1_tasks.append(qa_task.strip())
                context_list.append(context)
                qa_graph_index.append(index)
                qa_title_list.append(title)
                question_type += 1
                question_type = question_type % 3
        qa_s1_result = self.task_generator.inference(qa_s1_tasks, self.gen_system_prompt)
        qa_s2_tasks = []
        qa_graph_s2_index = []
        qa_graph_title_s2_index = []
        for qa_pairs, index, title, context in zip(qa_s1_result, qa_graph_index, qa_title_list, context_list):
            qa_pairs = qa_pairs["response"].strip()
            qa_pairs = qa_pairs.split(TUPLE_DELIMITER)
            if len(qa_pairs) == 1:
                qa_pairs = qa_pairs[0].split("\n\n")
            for qa_pair in qa_pairs:
                qa_pair = qa_pair.strip().split("\n")
                if len(qa_pair) != 3:
                    continue
                question_type, question, answer = qa_pair
                question = question.split(":")[-1].strip("\n").strip()
                answer = answer.split(":")[-1].strip("\n").strip()
                qa_task = self.gen_s2_user_prompt.format(context=context, question=question, answer=answer)
                qa_s2_tasks.append(qa_task.strip())
                qa_graph_s2_index.append(index)
                qa_graph_title_s2_index.append(title)
        qa_s2_result = self.task_generator.inference(qa_s2_tasks, self.gen_system_prompt)
        task_gen_results = [[] for _ in range(len(self.graph_list))]
        for qa_pair, index, title in zip(qa_s2_result, qa_graph_s2_index, qa_graph_title_s2_index):
            qa_pair = qa_pair["response"].strip()
            qa_pair = qa_pair.strip().split("\n")
            if len(qa_pair) != 3:
                continue
            question_type, question, answer = qa_pair
            question = question.split(":")[-1].strip("\n").strip()
            answer = answer.split(":")[-1].strip("\n").strip()
            answer = self.answer_format.format(answer=answer)
            user_prompt = self.template.format(
                title=title,
                question=question,
            )
            qa_train_text = self.create_chat_message(user_prompt, answer)
            task_gen_results[index].append(qa_train_text)
        return self.sample_post_process(task_gen_results)
