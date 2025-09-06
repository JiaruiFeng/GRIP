import numpy as np
from transformers import PreTrainedTokenizer

from grip.tasks.utils import shuffle_graph, clean_graph
from .gen_base import GenGraphTaskBase


class GenGraphContextTask(GenGraphTaskBase):
    edge_context_format = "source node: {src}, target node: {tgt}, relation: {rel}."

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

        train_context_result = []
        for graph, title in zip(self.graph_list, self.title_list):
            sample_train_context = []
            node_list = graph["node_list"]
            edge_list = graph["edge_list"]
            edge_index = np.array(graph["edge_index"])

            # clean graph, remove repeated edges and self-loops
            edge_list, edge_index = clean_graph(edge_list, edge_index)

            # random shuffle the graph order
            node_list, edge_list, edge_index = shuffle_graph(node_list, edge_list, edge_index)

            # plain edge information
            sample = f"The graph titled {title} contains edge: \n"
            for edge in edge_list:
                edge_text = self.edge_context_format.format(src=edge[0], tgt=edge[2], rel=edge[1])
                sample_train_context.append(sample + edge_text)
            train_context_result.append(sample_train_context)
        return train_context_result
