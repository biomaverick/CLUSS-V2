"""
clustering/cluster_extractor.py
================================
Extract final protein clusters from the phylogenetic tree given the
set of low co-similarity (cut) nodes.

Implements Stage 3 Step 4 from Kelil et al. (2007):
  A subtree is a VALID CLUSTER if:
    1. It contains NO low co-similarity node inside it.
    2. Its parent DOES contain a low co-similarity node
       (or there are no cut points at all -- entire tree is one cluster).

Sequences not assigned to any cluster of size >= min_size are orphans.

Fix #1 (iterative traversals):
  collect_leaves() and _traverse() were recursive and would crash with
  RecursionError on trees deeper than sys.getrecursionlimit() (~1 000).
  Both are now fully iterative.

Fix #3 (logging):
  Replaced print() calls with log.info() so output respects --log-level
  and HPC scheduler log capture.
"""

import logging
log = logging.getLogger(__name__)


from tree.phylo_tree import TreeNode


def collect_leaves(node: TreeNode) -> list[str]:
    """Collect all seq_id values in the subtree via iterative DFS."""
    leaves: list[str] = []
    stack: list = [node]
    while stack:
        n = stack.pop()
        if n.is_leaf:
            leaves.append(n.seq_id)
        else:
            for child in n.children:
                stack.append(child)
    return leaves


def extract_clusters(root: TreeNode,
                     cut_nodes: set[int],
                     min_size: int = 2
                     ) -> tuple[list[list[str]], list[str], dict]:
    """
    Extract cluster membership from the tree given cut node positions.

    Parameters
    ----------
    root      : root of the phylogenetic tree
    cut_nodes : set of node IDs identified as low co-similarity
    min_size  : minimum cluster size; smaller groups -> orphans (default 2)

    Returns
    -------
    clusters    : list of lists, each inner list = seq_ids in one cluster
    orphans     : list of seq_ids not in any cluster of size >= min_size
    cluster_ids : dict[seq_id -> cluster_index (int) or 'orphan' (str)]
    """
    raw_clusters: list[list[str]] = []

    # Iterative post-order traversal that mirrors the original recursive logic.
    # Each stack item: (node, parent_is_cut)
    # We track, per node, whether its subtree contains a cut node by
    # building a bottom-up result map.
    subtree_has_cut: dict[int, bool] = {}

    # Collect nodes in reverse-postorder (DFS pre-order), then process
    # them bottom-up.
    visit_order: list = []
    stack: list = [root]
    while stack:
        n = stack.pop()
        visit_order.append(n)
        for child in n.children:
            stack.append(child)

    for node in reversed(visit_order):
        if node.is_leaf:
            subtree_has_cut[node.id] = False
            continue

        left_has_cut  = subtree_has_cut.get(node.children[0].id, False)
        right_has_cut = subtree_has_cut.get(node.children[1].id, False)

        if node.id in cut_nodes:
            # Children with no cut nodes inside are valid clusters.
            if not left_has_cut:
                raw_clusters.append(collect_leaves(node.children[0]))
            if not right_has_cut:
                raw_clusters.append(collect_leaves(node.children[1]))
            subtree_has_cut[node.id] = True
        else:
            subtree_has_cut[node.id] = left_has_cut or right_has_cut

    if not subtree_has_cut.get(root.id, False):
        # No cut points -> entire tree is one cluster
        raw_clusters.append(collect_leaves(root))

    # Separate into valid clusters and orphans
    clusters: list[list[str]] = []
    orphans:  list[str]       = []

    for group in raw_clusters:
        if len(group) >= min_size:
            clusters.append(group)
        else:
            orphans.extend(group)

    # Build reverse lookup
    cluster_ids: dict = {}
    for idx, cluster in enumerate(clusters):
        for seq_id in cluster:
            cluster_ids[seq_id] = idx
    for seq_id in orphans:
        cluster_ids[seq_id] = "orphan"

    # Fix #3: use log.info instead of print() so --log-level is respected
    log.info("Clusters extracted: %d clusters | %d orphans",
             len(clusters), len(orphans))
    for i, c in enumerate(clusters):
        log.debug("  Cluster %3d : %4d sequences", i, len(c))
    if orphans:
        log.debug("  Orphans     : %4d sequences", len(orphans))

    return clusters, orphans, cluster_ids
