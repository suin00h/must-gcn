"""Graph utilities — same conventions as CTR-GCN's `graph/tools.py`."""

import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, axis=0)
    n = A.shape[0]
    Dn = np.zeros((n, n))
    for i in range(n):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** -1
    return A @ Dn


def get_spatial_graph(num_node, self_link, inward, outward):
    """3-partition spatial graph: identity / inward / outward."""
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    return np.stack((I, In, Out))
