import numpy as np
from transformers import PreTrainedTokenizer

from constants import SYSTEM_PROMPT, ANSWER_TAG
from grip.tasks.utils import sample_fixed_hop_size_neighbor, clean_graph, shuffle_graph
from .gen_base import GenGraphTaskBase


class GenSummarizationTask(GenGraphTaskBase):
    gen_system_prompt = SYSTEM_PROMPT

    gen_user_s1_prompt = """
    You are given a subgraph centered on a root node with its direct neighbors, representing real-world facts. Your 
    task is to generate one summary that capture all information in the subgraph. The summary should: 
    - Focus on the relations between the root node and its neighbors, including multi-entity connections (e.g., triangle relations).
    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.
    - rewrite the original context in a completely different way, tone, writing style, and logic to summarize the original context.
    - Some rewriting techniques can be used, including reordering, exchanging active and passive sentences, or synonym replacement.
    The subgraph is provided below:
    {context}
    
    Please answer in the following format: \nSummary: [summary]. 
    Please DON'T output quotes. 
    """

    gen_user_s2_prompt = """
    You are given a subgraph centered on a root node with its direct neighbors, representing real-world facts, along 
    with a summary written from it. Write a new summary that:

    - Retains every correct fact from the provided summary.
    - Adds any facts the provided summary missed and corrects any inaccuracies, using the subgraph as the source of truth.
    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.
    - Differs maximally in wording and structure from the original (e.g., reordering, active↔passive, synonym choice).
    - If the summary is not provided, directly generate a summary from the given subgraph.
    The subgraph is provided below:
    {context}
    
    The summary is provided below:
    {summary}

    Please answer in the following format: \nSummary: [summary]. 
    Please DON'T output quotes. 
    """

    answer_format = f"<{ANSWER_TAG}>" + "{answer}" + f"</{ANSWER_TAG}>"

    template = """Given the context graph titled {title}, please summarize the information surrounding the 
            {root_node}."""

    def __init__(
            self,
            graph_list: list,
            title_list: list[str],
            tokenizer: PreTrainedTokenizer,
            task_generator_model_name: str = "qwen-32b",
            task_gen_max_length: int = 1000,
            num_summarization: int = 10,
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
        self.num_summarization = num_summarization

    def gen_task(self, gen_empty_task=False) -> list:
        if gen_empty_task or self.num_summarization <= 0:
            return [[] for _ in range(len(self.graph_list))]
        # generate question and answer
        summary_s1_tasks = []
        graph_index = []
        root_node_list = []
        summary_title_list = []
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

            # stage 1. generate initial summaries
            for _ in range(self.num_summarization):
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
                # sample task type
                summary_task = self.gen_user_s1_prompt.format(context=context)
                summary_s1_tasks.append(summary_task.strip())
                graph_index.append(index)
                root_node_list.append(node_list[root_node])
                summary_title_list.append(title)
                context_list.append(context)
        summary_s1_result = self.task_generator.inference(summary_s1_tasks, self.gen_system_prompt)
        summary_s1_list = []
        for summary, index, root_node, title in zip(summary_s1_result, graph_index, root_node_list, summary_title_list):
            summary = summary["response"]
            summary = summary.split("Summary:")[-1].strip("\n").strip()
            summary_s1_list.append(summary)

        summary_s2_tasks = []
        for context, summary in zip(context_list, summary_s1_list):
            # stage 2. refine the initial summaries
            summary_task = self.gen_user_s2_prompt.format(context=context, summary=summary)
            summary_s2_tasks.append(summary_task.strip())
        summary_s2_result = self.task_generator.inference(summary_s2_tasks, self.gen_system_prompt)
        task_gen_results = [[] for _ in range(len(self.graph_list))]
        for summary1, summary2, index, root_node, title in zip(summary_s1_result, summary_s2_result, graph_index,
                                                               root_node_list, summary_title_list):
            summary1 = summary1["response"]
            summary1 = summary1.split("Summary:")[-1].strip("\n").strip()
            if len(summary1) > 10:
                summary1 = self.answer_format.format(answer=summary1)
                sample = self.template.format(
                    title=title,
                    root_node=root_node,
                )
                sample = self.create_chat_message(sample, summary1)
                task_gen_results[index].append(sample)

            summary2 = summary2["response"]
            summary2 = summary2.split("Summary:")[-1].strip("\n").strip()
            if len(summary2) > 10:
                summary2 = self.answer_format.format(answer=summary2)
                sample = self.template.format(
                    title=title,
                    root_node=root_node,
                )
                sample = self.create_chat_message(sample, summary2)
                task_gen_results[index].append(sample)

        return self.sample_post_process(task_gen_results)
