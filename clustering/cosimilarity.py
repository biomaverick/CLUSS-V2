"""
clustering/cosimilarity.py
══════════════════════════
Compute leaf weights (Thompson 1994 method) and co-similarity values
(Ward variant, Batagelj formulation) for all nodes in the phylogenetic tree.
These are Stage 3 Steps 1 and 2 from Kelil et al. (2007).
No algorithmic changes — these were already correct.
Refactored for clarity, explicit type hints, and iterative traversals.
"""

import logging
log = logging.getLogger(__name__)


from tree.phylo_tree import TreeNode


def compute_leaf_weights(root: TreeNode) -> dict[int, float]:
    """
    Compute Thompson sequence weights for all leaf nodes.
    Formula (Thompson et al. 1994, Comput Appl Biosci 10:19-29):
        W_L = Sum  D_{Parent(i), i} / d_{Parent(i)}
              for all i on the path from L to root (excluding root)
    Where:
      D_{Parent(i), i} = branch length from node i to its parent
      d_{Parent(i)}    = number of leaves under i's parent
    Interpretation:
      Small W_L -> highly representative sequence (many close relatives)
      Large W_L -> isolated, poorly represented sequence
    Parameters
    ----------
    root : root TreeNode of the phylogenetic tree
    Returns
    -------
    dict[leaf_node_id -> weight (float)]
    """
    leaf_weights: dict[int, float] = {}

    # Iterative DFS -- avoids RecursionError on deep (unbalanced) trees.
    # Stack items: (node, accumulated_weight_so_far)
    stack: list[tuple] = [(root, 0.0)]
    while stack:
        node, accumulated = stack.pop()
        if node.is_leaf:
            leaf_weights[node.id] = accumulated
            continue
        for child in node.children:
            contribution = (child.branch_length / node.n_leaves
                           if node.n_leaves > 0 else 0.0)
            stack.append((child, accumulated + contribution))

    return leaf_weights


def compute_node_weights(root: TreeNode,
                         leaf_weights: dict[int, float]) -> dict[int, float]:
    """
    Compute node weights for all nodes (leaf and internal).
    For leaves: node_weight[leaf] = leaf_weights[leaf]
    For internal nodes: W_P = W_L + W_R  (sum of children's weights)
    Computed via postorder (bottom-up) traversal.
    Parameters
    ----------
    root         : root TreeNode
    leaf_weights : output of compute_leaf_weights()
    Returns
    -------
    dict[node_id -> weight (float)]
    """
    node_weights: dict[int, float] = {}

    # Iterative postorder (children before parent).
    # Pass 1: collect nodes in reverse-postorder via DFS.
    order: list = []
    stack: list = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        for child in node.children:
            stack.append(child)

    # Pass 2: process in reverse (leaves -> root).
    for node in reversed(order):
        if node.is_leaf:
            node_weights[node.id] = leaf_weights.get(node.id, 0.0)
        else:
            node_weights[node.id] = sum(
                node_weights.get(c.id, 0.0) for c in node.children
            )

    return node_weights


def compute_cosimilarity(root: TreeNode,
                         node_weights: dict[int, float]) -> dict[int, float]:
    """
    Compute co-similarity for every internal node P.
    Formula (generalised Ward / Batagelj):
        C_P = (W_L * W_R) / (W_L + W_R) * S_{L,R}
    Where:
      W_L, W_R  = weights of left and right child subtrees
      S_{L,R}   = similarity at time of merge (stored on node P)
    Interpretation:
      C_P is inversely proportional to within-cluster variance.
      Large C_P -> tight, compact cluster; Small C_P -> candidate cut point.
    Parameters
    ----------
    root         : root TreeNode
    node_weights : output of compute_node_weights()
    Returns
    -------
    dict[internal_node_id -> co-similarity (float)]
    """
    cosim: dict[int, float] = {}

    # Iterative preorder -- any traversal order works here because
    # co-similarity at P depends only on its direct children's weights,
    # which are already fully computed in node_weights.
    stack: list = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            continue

        L, R  = node.children[0], node.children[1]
        W_L   = node_weights.get(L.id, 0.0)
        W_R   = node_weights.get(R.id, 0.0)
        S_LR  = (node.merge_similarity
                 if node.merge_similarity is not None else 0.0)

        denom = W_L + W_R
        cosim[node.id] = (W_L * W_R) / denom * S_LR if denom > 0 else 0.0

        stack.append(L)
        stack.append(R)

    return cosim
