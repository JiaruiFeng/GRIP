import random
from typing import Union, Optional

import numpy as np
from scipy.sparse import csr_array, csr_matrix

from constants import (
    NODE_TAG,
    EDGE_TAG,
    GRAPH_TAG,
)

import string
import torch


def edge_list_to_sequence(edge_list: list[list[str]], shuffle: bool = True) -> str:
    seq = f"<{GRAPH_TAG}>"
    if shuffle:
        random.shuffle(edge_list)
    for edge in edge_list:
        s, r, t = edge
        seq += f"<{NODE_TAG}>" + s + f"</{NODE_TAG}>"
        seq += f"<{EDGE_TAG}>" + r + f"</{EDGE_TAG}>"
        seq += f"<{NODE_TAG}>" + t + f"</{NODE_TAG}>"
        seq += "\n"
    return seq + f"</{GRAPH_TAG}>"


def edge_with_index_to_sequence(
        node_list: list[str],
        edge_list: list[list[str]],
        edge_index: list[list],
        shuffle: bool = True
) -> str:
    seq = f"<{GRAPH_TAG}>"
    node_seq = ""
    node_ids = [i for i in range(len(node_list))]
    for i, n in enumerate(node_list):
        node_seq += f"<{NODE_TAG}>" + f"ID: {node_ids[i]}. " + n + f"</{NODE_TAG}>" + "\n"

    if shuffle:
        order = np.random.permutation(len(edge_index))
        edge_index = [edge_index[i] for i in order]
    edge_seq = ""
    for i, e in enumerate(edge_index):
        s_id, t_id = e
        r = edge_list[i][1]
        edge_seq += (f"<{EDGE_TAG}>" + f"Node {node_ids[s_id]} and node {node_ids[t_id]} has relation "
                     + r + f"</{EDGE_TAG}>" + "\n")

    seq = seq + node_seq + edge_seq + f"</{GRAPH_TAG}>"
    return seq


def clean_graph(edge_list: list[list], edge_index: np.ndarray) -> tuple[list, np.ndarray]:
    # remove repeated edges and self-loops
    result_edge_list = []
    result_edge_index = []
    edge_set = set()
    for src, tgt in edge_index:
        if src == tgt:
            continue
        edge = (src, tgt)
        if edge not in edge_set:
            edge_set.add(edge)
            result_edge_list.append(edge_list[len(result_edge_index)])
            result_edge_index.append(edge)
    return result_edge_list, np.array(result_edge_index)


def shuffle_graph(
        node_list: list[str],
        edge_list: list[list[str]],
        edge_index: np.ndarray,
) -> tuple[list[str], list[list[str]], np.ndarray]:
    # random shuffle the graph order
    node_indices = np.arange(len(node_list))
    np.random.shuffle(node_indices)
    inverse_node_indices = np.argsort(node_indices)
    node_list = [node_list[i] for i in node_indices]
    edge_index[:, 0] = inverse_node_indices[edge_index[:, 0]]
    edge_index[:, 1] = inverse_node_indices[edge_index[:, 1]]

    edge_indices = np.arange(len(edge_list))
    np.random.shuffle(edge_indices)
    edge_list = [edge_list[i] for i in edge_indices]
    edge_index = edge_index[edge_indices]
    return node_list, edge_list, edge_index


def sample_fixed_hop_size_neighbor(
        edge_index: Union[np.ndarray, csr_matrix],
        edge_indices: np.ndarray,
        root: Union[int, list],
        num_nodes: Optional[int] = None,
        hop: int = 3,
        max_nodes_per_hop: int = 5) -> tuple[np.ndarray, csr_matrix]:
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    if isinstance(edge_index, np.ndarray):
        adj_mat = csr_array((edge_indices, (edge_index[0], edge_index[1]),),
                            shape=(num_nodes, num_nodes), )
    else:
        adj_mat = edge_index
    if isinstance(root, int):
        root = [root]
    visited = np.array(root)
    fringe = np.array(root)
    nodes = np.array(root)
    for h in range(1, hop + 1):
        u = adj_mat[fringe].nonzero()[1]
        fringe = np.setdiff1d(u, visited)
        visited = np.union1d(visited, fringe)
        if len(fringe) > max_nodes_per_hop:
            fringe = np.random.choice(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = np.concatenate([nodes, fringe])
        # dist_list+=[dist+1]*len(fringe)
    nodes = nodes.astype(int)
    edges = adj_mat[nodes, :][:, nodes]
    return nodes, edges
