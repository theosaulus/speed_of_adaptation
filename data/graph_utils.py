"""
Copyright (c) 2021, QUVA-Lab, University of Amsterdam
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.

------------------------------------------------------------

Utility functions for the graphs, such as edge<->adj_matrix conversion
and sorting variables according to the global causal order.
"""
import numpy as np


def adj_matrix_to_edges(adj_matrix):
    """
    Converts an adjacency matrix to a list of edges.
    """
    edges = np.where(adj_matrix)
    edges = np.stack([edges[0], edges[1]], axis=1)
    return edges


def edges_to_adj_matrix(edges, num_vars):
    """
    Converts a list of edges to an adjacency matrix
    """
    if not isinstance(edges, np.ndarray):
        edges = np.array(edges)
    adj_matrix = np.zeros((num_vars, num_vars), dtype=np.bool)
    if edges.shape[0] > 0:
        adj_matrix[edges[:, 0], edges[:, 1]] = True
    return adj_matrix


def edges_or_adj_matrix(edges, adj_matrix, num_vars):
    """
    Converts edges to adjacency matrix, or vice versa depending on which of the two is given and which is None.
    """
    assert edges is not None or adj_matrix is not None, 'Either the edges or adjacency matrix must be provided for the DAG'
    if edges is None:
        edges = adj_matrix_to_edges(adj_matrix)
    elif not isinstance(edges, np.ndarray):
        edges = np.array(edges)
    if adj_matrix is None:
        adj_matrix = edges_to_adj_matrix(edges, num_vars)
    return edges, adj_matrix


def sort_graph_by_vars(variables, edges=None, adj_matrix=None, latents=None):
    """
    Takes a list of variables and graph structure, and determines the causal order of the variable, 
    i.e., an order in which we can perform ancestral sampling. Returns the newly sorted graph structure.
    """
    edges, adj_matrix = edges_or_adj_matrix(edges, adj_matrix, len(variables))
    matrix_copy = np.copy(adj_matrix)

    sorted_idxs = []

    def get_empty_nodes():
        return [i for i in np.where(~matrix_copy.any(axis=0))[0] if i not in sorted_idxs]

    if latents is None or latents.shape[0] == 0 or latents[0, 0] < 0:
        empty_nodes = get_empty_nodes()
    else:
        empty_nodes = [latents[i, 0] for i in range(latents.shape[0])]
    while len(empty_nodes) > 0:
        node = empty_nodes.pop(0)
        sorted_idxs.append(node)
        matrix_copy[node, :] = False
        empty_nodes = get_empty_nodes()
    assert not matrix_copy.any(), "Sorting the graph failed because it is not a DAG!"

    variables = [variables[i] for i in sorted_idxs]
    adj_matrix = adj_matrix[sorted_idxs][:, sorted_idxs]

    num_vars = len(variables)
    edges = edges - num_vars  # To have a better replacement
    if latents is not None:
        latents = latents - num_vars
    for v_idx, n_idx in enumerate(sorted_idxs):
        edges[edges == (n_idx - num_vars)] = v_idx
        if latents is not None:
            latents[latents == (n_idx - num_vars)] = v_idx

    if latents is not None:
        latents[:, 1:] = np.sort(latents[:, 1:], axis=-1)
        return variables, edges, adj_matrix, latents, sorted_idxs
    else:
        return variables, edges, adj_matrix, sorted_idxs


def get_node_relations(adj_matrix):
    """
    Returns a matrix which describes the relations fo each node pair beyond parent-child relations.

    Parameters
    ----------
    adj_matrix : np.ndarray, shape [num_vars, num_vars], type np.bool
                 The adjacency matrix of the graph.

    Returns
    -------
    node_relations : np.ndarray, shape [num_vars, num_vars], type np.int32
                     A matrix, where an element (i,j) can take the following values:
                       node_relations[i,j] = 1: j is an ancestor of i
                       node_relations[i,j] = -1: j is a descendant of i,
                       node_relations[i,j] = 0: j and i are independent conditioned on the empty set
                       node_relations[i,j] = 2: j and i share a confounder
    global_roles : np.ndarray of str, shape [num_vars]
        For each node: 'root' (no incoming edges), 'leaf' (no outgoing edges), or 'neither'.
    in_degrees : np.ndarray of int, shape [num_vars]
        Number of incoming edges per node.
    out_degrees : np.ndarray of int, shape [num_vars]
        Number of outgoing edges per node.
    total_degrees : np.ndarray of int, shape [num_vars]
        Sum of incoming and outgoing edges per node.    
    """
    # Find all ancestor-descendant relations
    ancestors = adj_matrix.T
    changed = True
    while changed: 
        new_anc = np.logical_and(ancestors[..., None], ancestors[None]).any(axis=1)
        new_anc = np.logical_or(ancestors, new_anc)
        changed = not (new_anc == ancestors).all().item()
        ancestors = new_anc

    # Output: matrix with (i,j)
    #         = 1: j is an ancestor of i
    #         = -1: j is a descendant of i,
    #         = 0: j and i are independent
    #         = 2: j and i share a confounder
    ancestors = ancestors.astype(np.int32)
    descendant = ancestors.T
    node_relations = ancestors - descendant
    confounder = (node_relations == 0) * ((ancestors[None] * ancestors[:, None]).sum(axis=-1) > 0)
    node_relations += 2 * confounder
    node_relations[np.arange(node_relations.shape[0]), np.arange(node_relations.shape[1])] = 0
    # One-hop relations matrix: parent if 1, child if -1
    one_hop_relations = - adj_matrix.astype(np.int32) + adj_matrix.T.astype(np.int32)

    # in, out, and total degrees
    in_degrees  = adj_matrix.sum(axis=0).astype(int)
    out_degrees = adj_matrix.sum(axis=1).astype(int)
    total_degrees = in_degrees + out_degrees

    # global role: root / leaf / neither
    global_roles = np.full(adj_matrix.shape[0], 'neither', dtype=object)
    global_roles[(in_degrees == 0) & (out_degrees > 0)] = 'root'
    global_roles[(out_degrees == 0) & (in_degrees > 0)] = 'leaf'

    return node_relations, one_hop_relations, global_roles, in_degrees, out_degrees, total_degrees