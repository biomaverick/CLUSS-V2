"""
tree/phylo_tree.py
══════════════════
Hierarchical phylogenetic tree construction from a similarity matrix.

Two tree-building methods are implemented (MD Section 2.4 and 3.2):

1. UPGMA (default, matches original paper)
   ─────────────────────────────────────────────────────────────────
   Weighted average-linkage hierarchical clustering.
   Assumes the molecular clock (all lineages evolve at the same rate).
   Produces an ultrametric tree — appropriate when evolutionary rates
   are approximately equal across the protein family.

2. Neighbor Joining (--tree-method nj)
   ─────────────────────────────────────────────────────────────────
   Saitou & Nei (1987) — does NOT assume molecular clock.
   More accurate for divergent sequences where evolutionary rates
   vary across lineages (e.g., surface loops vs. catalytic cores).
   Recommended for families spanning large evolutionary distances.

   The Q-criterion selects merges that minimise total branch length,
   compensating for rate variation.

Bug fix applied
───────────────
  TreeNode.depth was missing, causing feature 3 of the RF boundary
  detector to always return 0. Fixed: depth is assigned via
  assign_depths() after tree construction completes.

References
──────────
Saitou & Nei (1987) Mol Biol Evol 4:406–425.
Kelil et al. (2007) BMC Bioinformatics 8:286.
"""

import logging
log = logging.getLogger(__name__)


import numpy as np
import heapq


# ─────────────────────────────────────────────────────────────────────────────
# Tree node
# ─────────────────────────────────────────────────────────────────────────────

class TreeNode:
    """
    Node in the hierarchical phylogenetic tree.

    Attributes
    ----------
    id               : unique integer node ID
    is_leaf          : True for leaf (single sequence) nodes
    seq_id           : sequence identifier string (leaves only)
    children         : list of child TreeNode objects
    parent           : parent TreeNode (None for root)
    branch_length    : evolutionary distance from this node to its parent
    n_leaves         : number of leaf descendants
    merge_similarity : S_{L,R} similarity at the time this node was created
    depth            : depth in tree (root = 0) — FIXED: was missing
    bootstrap        : bootstrap support fraction [0, 1] (set externally)
    """

    __slots__ = (
        "id", "is_leaf", "seq_id", "children", "parent",
        "branch_length", "n_leaves", "merge_similarity",
        "depth", "bootstrap"
    )

    def __init__(self, node_id: int,
                 is_leaf: bool = True,
                 seq_id: str | None = None):
        self.id               = node_id
        self.is_leaf          = is_leaf
        self.seq_id           = seq_id
        self.children:  list  = []
        self.parent:    TreeNode | None = None
        self.branch_length    = 0.0
        self.n_leaves         = 1
        self.merge_similarity = None
        self.depth            = 0       # FIXED: initialised; set by assign_depths()
        self.bootstrap        = None


def assign_depths(root: "TreeNode") -> None:
    """
    Assign depth values to all nodes via iterative BFS.
    Must be called once after tree construction is complete.
    """
    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        node.depth = d
        for child in node.children:
            stack.append((child, d + 1))


# ─────────────────────────────────────────────────────────────────────────────
# UPGMA tree (original paper method)
# ─────────────────────────────────────────────────────────────────────────────

def build_upgma_tree(S: np.ndarray,
                     seq_ids: list[str]) -> tuple[TreeNode, dict]:
    """
    Build UPGMA hierarchical tree from similarity matrix S.

    Algorithm (paper Stage 2, Kelil et al. 2007):
      - Each sequence starts as its own leaf node.
      - Iteratively merge the two most similar active nodes L and R.
      - Similarity of new parent P to other nodes K:
          S_{P,K} = (dL * S_{L,K} + dR * S_{R,K}) / (dL + dR)
      - Branch lengths: D_{L,P} = D_{R,P} = (1 - S_{L,R}) / 2

    Uses a max-heap for O(N log N) merge selection instead of O(N²)
    submatrix scan, significantly faster for large N.

    Parameters
    ----------
    S       : N×N similarity matrix
    seq_ids : sequence ID list (same order as S rows/cols)

    Returns
    -------
    (root TreeNode, nodes dict[id -> TreeNode])
    """
    N = len(seq_ids)

    nodes:  dict[int, TreeNode] = {}
    for i in range(N):
        nodes[i] = TreeNode(i, is_leaf=True, seq_id=seq_ids[i])

    active  = set(range(N))
    # Extend similarity matrix in-place (will grow as merges happen)
    sim     = np.zeros((2 * N, 2 * N), dtype=np.float64)
    sim[:N, :N] = S

    next_id = N

    # Build initial max-heap (negate similarities for min-heap)
    heap: list[tuple[float, int, int]] = []
    for i in range(N):
        for j in range(i + 1, N):
            heapq.heappush(heap, (-S[i, j], i, j))

    while len(active) > 1:
        # Pop until we find a valid pair (both still active)
        while heap:
            neg_s, L_id, R_id = heapq.heappop(heap)
            if L_id in active and R_id in active:
                break
        else:
            break  # heap exhausted (shouldn't happen)

        S_LR = -neg_s
        L, R = nodes[L_id], nodes[R_id]
        dL, dR = L.n_leaves, R.n_leaves

        branch_len = (1.0 - S_LR) / 2.0

        P = TreeNode(next_id, is_leaf=False)
        P.children         = [L, R]
        P.n_leaves         = dL + dR
        P.merge_similarity = float(S_LR)

        L.parent = R.parent = P
        L.branch_length = R.branch_length = branch_len

        nodes[next_id] = P

        # Update similarities to new node P
        for K_id in active:
            if K_id in (L_id, R_id):
                continue
            s_pk = (dL * sim[L_id, K_id] + dR * sim[R_id, K_id]) / (dL + dR)
            sim[next_id, K_id] = sim[K_id, next_id] = s_pk
            heapq.heappush(heap, (-s_pk, min(next_id, K_id), max(next_id, K_id)))

        sim[next_id, next_id] = 1.0

        active.discard(L_id)
        active.discard(R_id)
        active.add(next_id)

        next_id += 1

    root = nodes[next(iter(active))]
    assign_depths(root)
    return root, nodes


# ─────────────────────────────────────────────────────────────────────────────
# Neighbor Joining tree (MD Section 2.4 / 3.2)
# ─────────────────────────────────────────────────────────────────────────────

def build_nj_tree(S: np.ndarray,
                  seq_ids: list[str]) -> tuple[TreeNode, dict]:
    """
    Build a Neighbor Joining tree from similarity matrix S.

    NJ does not assume the molecular clock (contrast with UPGMA), making
    it statistically consistent under a much wider class of evolutionary
    models. Recommended for datasets spanning large evolutionary distances.

    Algorithm (Saitou & Nei 1987, Mol Biol Evol 4:406–425):
      D = 1 - S  (distance matrix)
      Q[i,j] = (n-2)*D[i,j] - R[i] - R[j]   where R[i] = Σ_k D[i,k]
      Select pair (i,j) minimising Q (not D directly).
      Branch lengths:
        bl_i = D[i,j]/2 + (R[i] - R[j]) / (2*(n-2))
        bl_j = D[i,j] - bl_i
      Update distance to new node u:
        D[u,k] = (D[i,k] + D[j,k] - D[i,j]) / 2

    Parameters
    ----------
    S       : N×N similarity matrix
    seq_ids : sequence ID list

    Returns
    -------
    (root TreeNode, nodes dict[id -> TreeNode])
    """
    N = len(seq_ids)

    nodes: dict[int, TreeNode] = {
        i: TreeNode(i, is_leaf=True, seq_id=seq_ids[i])
        for i in range(N)
    }

    # Work with a large-enough distance matrix
    max_size = 2 * N
    D = np.zeros((max_size, max_size), dtype=np.float64)
    D[:N, :N] = 1.0 - S
    np.fill_diagonal(D, 0.0)

    active  = list(range(N))
    next_id = N

    while len(active) > 2:
        n_act = len(active)

        # Vectorised Q-matrix: O(N²) numpy instead of O(N²) Python loops
        active_arr = np.array(active, dtype=np.int64)
        D_sub = D[np.ix_(active_arr, active_arr)]      # (n_act × n_act) submatrix
        R_vec = D_sub.sum(axis=1)                      # row sums

        # Q[a,b] = (n_act-2)*D[a,b] - R[a] - R[b]
        Q_mat = (n_act - 2) * D_sub - R_vec[:, None] - R_vec[None, :]
        np.fill_diagonal(Q_mat, np.inf)                # exclude self-pairs

        flat_idx = int(np.argmin(Q_mat))
        a_idx, b_idx = divmod(flat_idx, n_act)
        i = int(active_arr[a_idx])
        j = int(active_arr[b_idx])

        R = {int(active_arr[k]): float(R_vec[k]) for k in range(n_act)}

        # Branch lengths (clamped to ≥ 0 to handle rounding)
        denom = 2 * max(1, n_act - 2)
        bl_i  = max(0.0, D[i, j] / 2.0 + (R[i] - R[j]) / denom)
        bl_j  = max(0.0, D[i, j] - bl_i)

        P = TreeNode(next_id, is_leaf=False)
        for child, bl in [(nodes[i], bl_i), (nodes[j], bl_j)]:
            child.branch_length = bl
            child.parent        = P
            P.children.append(child)

        P.n_leaves         = nodes[i].n_leaves + nodes[j].n_leaves
        P.merge_similarity = float(max(0.0, 1.0 - D[i, j]))

        # Update distances to new node
        for k in active:
            if k in (i, j):
                continue
            d_new = (D[i, k] + D[j, k] - D[i, j]) / 2.0
            D[next_id, k] = D[k, next_id] = max(0.0, d_new)

        D[next_id, next_id] = 0.0
        nodes[next_id] = P

        active = [x for x in active if x not in (i, j)] + [next_id]
        next_id += 1

    # Join the last two nodes
    ii, jj = active[0], active[1]
    P = TreeNode(next_id, is_leaf=False)
    for child in [nodes[ii], nodes[jj]]:
        child.branch_length = max(0.0, D[ii, jj] / 2.0)
        child.parent        = P
        P.children.append(child)
    P.n_leaves         = nodes[ii].n_leaves + nodes[jj].n_leaves
    P.merge_similarity = float(max(0.0, 1.0 - D[ii, jj]))
    nodes[next_id]     = P

    assign_depths(P)
    return P, nodes


# ─────────────────────────────────────────────────────────────────────────────
# Unified builder
# ─────────────────────────────────────────────────────────────────────────────

def build_phylo_tree(S: np.ndarray,
                     seq_ids: list[str],
                     method: str = "upgma") -> tuple[TreeNode, dict]:
    """
    Build a phylogenetic tree from similarity matrix S.

    Parameters
    ----------
    S       : N×N similarity matrix
    seq_ids : sequence ID list
    method  : 'upgma' (default, matches paper) or 'nj' (Neighbor Joining)

    Returns
    -------
    (root TreeNode, nodes dict[id -> TreeNode])
    """
    method = method.lower()
    if method == "nj":
        log.info("  Tree method: Neighbor Joining (no molecular clock assumption)")
        return build_nj_tree(S, seq_ids)
    else:
        log.info("  Tree method: UPGMA (weighted average linkage)")
        return build_upgma_tree(S, seq_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Newick serialisation
# ─────────────────────────────────────────────────────────────────────────────

def to_newick(node: TreeNode,
              bootstrap: dict | None = None) -> str:
    """
    Serialise tree to Newick format string.

    Parameters
    ----------
    node      : root TreeNode
    bootstrap : optional dict[node_id -> support_fraction (0–1)]
                Internal nodes annotated as (Child1,Child2)90:0.1234
                where 90 = 90% bootstrap support.

    Returns
    -------
    Newick string (no trailing newline).
    """
    if node.is_leaf:
        return f"{node.seq_id}:{node.branch_length:.6f}"

    children_str = ",".join(to_newick(c, bootstrap) for c in node.children)

    sup = ""
    if bootstrap is not None and node.id in bootstrap:
        sup = str(int(round(bootstrap[node.id] * 100)))

    return f"({children_str}){sup}:{node.branch_length:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# Newick round-trip: load Newick → TreeNode graph
# ─────────────────────────────────────────────────────────────────────────────

def newick_to_treenodes(newick_path: str) -> tuple["TreeNode", dict]:
    """
    Reconstruct a (root, nodes) pair from a Newick file on disk.

    Uses BioPython's ``Phylo.read`` (already a project dependency) and
    maps each clade to a ``TreeNode`` so downstream stages (co-similarity,
    cluster extraction, etc.) work unchanged.

    Parameters
    ----------
    newick_path : path to the ``.nwk`` file written by ``_save_tree``

    Returns
    -------
    (root TreeNode, nodes dict[int -> TreeNode])
    """
    from Bio import Phylo
    import io

    with open(newick_path) as fh:
        newick_str = fh.read().strip()

    bio_tree = Phylo.read(io.StringIO(newick_str), "newick")
    nodes: dict[int, TreeNode] = {}
    _id_counter = [0]

    def _convert(clade, parent=None) -> TreeNode:
        node_id = _id_counter[0]
        _id_counter[0] += 1

        is_leaf = (clade.name is not None and not clade.clades)
        seq_id  = clade.name if is_leaf else None
        node    = TreeNode(node_id, is_leaf=is_leaf, seq_id=seq_id)
        node.branch_length = float(clade.branch_length or 0.0)
        node.parent        = parent
        nodes[node_id]     = node

        for child_clade in clade.clades:
            child = _convert(child_clade, parent=node)
            node.children.append(child)
            node.n_leaves += child.n_leaves

        if is_leaf:
            node.n_leaves = 1

        return node

    root = _convert(bio_tree.root)
    assign_depths(root)
    return root, nodes
