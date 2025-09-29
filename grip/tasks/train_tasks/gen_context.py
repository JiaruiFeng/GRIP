from typing import Optional

import numpy as np
from transformers import PreTrainedTokenizer

from constants import ANSWER_TAG, SYSTEM_PROMPT
from grip.tasks.utils import shuffle_graph, clean_graph, compute_node_edge_weight
from models import BaseInferenceModel
from .gen_base import GenGraphTaskBase


def renormalize_weights(weights, max_clip=100):
    """
    Renormalize a list of weights so that the minimum value is 1,
    convert to integers, and clip to a maximum value.
    Args:
        weights (list of float or int): The input weights.
        max_clip (int): The maximum allowed value for the normalized weights.
    Returns:
        list of int: The renormalized, integer, and clipped weights.
    """
    min_weight = min(weights)
    # Normalize so minimum is 1
    normalized = [w / min_weight for w in weights]
    # Convert to integer (using round, can use int() for floor)
    int_normalized = [int(round(w)) for w in normalized]
    # Clip to max_clip
    clipped = [min(w, max_clip) for w in int_normalized]
    return clipped

class GenGraphContextTask(GenGraphTaskBase):

    node_context_format = "In context graph {title}, {node}."

    edge_context_format = "In context graph {title}, {src} is {rel} {tgt}."


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
        for index, (graph, title) in enumerate(zip(self.graph_list, self.title_list)):
            sample_train_context = []
            node_list = graph["node_list"]
            edge_list = graph["edge_list"]
            edge_index = np.array(graph["edge_index"])

            # clean graph, remove repeated edges and self-loops
            edge_list, edge_index = clean_graph(edge_list, edge_index)

            # random shuffle the graph order
            node_list, edge_list, edge_index = shuffle_graph(node_list, edge_list, edge_index)

            num_node = len(node_list)
            node_weight, edge_weight = compute_node_edge_weight(edge_index, num_node)

            node_up_sampling = renormalize_weights(node_weight, 4)
            edge_up_sampling = renormalize_weights(edge_weight, 3)

            for node, up_sample_num in zip(node_list, node_up_sampling):
                node_text = self.node_context_format.format(node=node, title=title)
                for _ in range(up_sample_num):
                    sample_train_context.append(node_text)

            # plain edge information
            for edge, up_sample_num in zip(edge_list, edge_up_sampling):
                edge_text = self.edge_context_format.format(src=edge[0], tgt=edge[2], rel=edge[1], title=title)
                for _ in range(up_sample_num):
                    sample_train_context.append(edge_text)

            train_context_result.append(sample_train_context)

        return train_context_result
