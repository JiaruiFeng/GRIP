import numpy as np

from constants import SYSTEM_PROMPT, TUPLE_DELIMITER, ANSWER_TAG
from grip.tasks.utils import sample_fixed_hop_size_neighbor, clean_graph, shuffle_graph
from .gen_base import GenGraphTaskBase


class GenSummarizationTask(GenGraphTaskBase):
    gen_system_prompt = SYSTEM_PROMPT

    gen_user_prompt = """
    You are given a subgraph centered on a root node with its direct neighbors, representing real-world facts. Your 
    task is to generate TWO summaries that capture all information in the subgraph. Each summary should: 
    - Focus on the relations between the root node and its neighbors, including multi-entity connections (e.g., triangle relations).
    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.
    - Differ as much as possible from the other, using techniques such as reordering, active/passive voice changes, or synonym replacement.
    Please answer in the following format: \nSummary: [summary] {tuple_delimiter} \nSummary: [summary]. 
    Please DON'T output quotes and separate two summary by {tuple_delimiter}. The subgraph is provided below:
    {context}
    """

    answer_format = f"<{ANSWER_TAG}>" + "{answer}" + f"</{ANSWER_TAG}>"

    template = """Given the context graph titled {title}, please summarize the information surrounding the node {root_node}. Response in the following format:""" + answer_format.format(answer="[answer]")

    def gen_task(self, gen_empty_task=False) -> list:
        if gen_empty_task:
            return [[] for _ in range(len(self.graph_list))]
        self.load_model()
        # generate question and answer
        summary_tasks = []
        graph_index = []
        root_node_list = []
        summary_title_list = []
        for index, (graph, title) in enumerate(zip(self.graph_list, self.title_list)):
            edge_list = graph["edge_list"]
            node_list = graph["node_list"]
            edge_index = np.array(graph["edge_index"])

            # clean graph, remove repeated edges and self-loops
            edge_list, edge_index = clean_graph(edge_list, edge_index)

            # random shuffle the graph order
            node_list, edge_list, edge_index = shuffle_graph(node_list, edge_list, edge_index)
            num_nodes = len(node_list)
            for root_node in range(num_nodes):
                subgraph_nodes, subgraph_edges = sample_fixed_hop_size_neighbor(
                    edge_index.T,
                    np.arange(len(edge_list)),
                    root_node,
                    num_nodes,
                    hop=1,
                    max_nodes_per_hop=6,
                )
                subgraph_edges = [edge_list[e] for e in subgraph_edges.data]

                context = "The root node is {root}. The subgraph including edges: ".format(root=node_list[root_node])
                entry_format = "the source node {src} is {rel} the target node {tgt}; "
                for entry in subgraph_edges:
                    context += entry_format.format(src=entry[0], rel=entry[1], tgt=entry[2])
                # sample task type
                summary_task = self.gen_user_prompt.format(context=context, tuple_delimiter=TUPLE_DELIMITER)
                summary_tasks.append(summary_task.strip())
                graph_index.append(index)
                root_node_list.append(node_list[root_node])
                summary_title_list.append(title)
        summary_result = self.task_generator.inference(summary_tasks, self.gen_system_prompt)
        task_gen_results = [[] for _ in range(len(self.graph_list))]
        for summaries, index, root_node, title in zip(summary_result, graph_index, root_node_list, summary_title_list):
            summaries = summaries["response"]
            summaries = summaries.split(TUPLE_DELIMITER)
            summaries = [summ for summ in summaries if summ]
            if len(summaries) == 1:
                summaries = summaries[0].split("\n\n")
            for summary in summaries:
                summary = summary.split("Summary:")[-1].strip("\n").strip()
                if len(summary) < 10:
                    continue
                summary = self.answer_format.format(answer=summary)
                sample = self.template.format(
                    title=title,
                    root_node=root_node,
                )
                sample = self.create_chat_message(sample, summary)
                task_gen_results[index].append(sample)
        self.unload_model()
        return self.sample_post_process(task_gen_results)
