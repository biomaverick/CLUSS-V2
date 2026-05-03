"""
similarity/sms_matrix.py
══════════════════════════
Build the full N×N pairwise SMS similarity matrix.

Key fixes and upgrades
──────────────────────
  FIX (MD Section 2.3): compute_s_max now returns the TOTAL weight of the
  longest sequence (not per-residue average).

  UPGRADE: multi-matrix blend (BLOSUM45/62/80) with configurable weights.

  UPGRADE: np.memmap-backed matrix for large N (avoids OOM on 30k+ sequences).
  Controlled by the chunk_size and out_dir parameters.

  UPGRADE: optional domain-aware scoring via InterPro domain annotations.

  UPGRADE: property-group pass uses Murphy 2000 reduced alphabets.
"""

import logging
import os
import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from tqdm import tqdm

from cluss_plus.similarity.sms_engine import (encode_sequence, compute_sms_pair)

log = logging.getLogger(__name__)

DEFAULT_MATRICES = ["BLOSUM45", "BLOSUM62", "BLOSUM80"]
DEFAULT_WEIGHTS  = [0.20,       0.60,       0.20]

# Memory threshold (bytes) above which we switch to np.memmap backing.
# 400 MB = 10k sequences at float32.
_MEMMAP_THRESHOLD_BYTES = 400 * 1024 * 1024


# ─────────────────────────────────────────────────────────────────────────────
# Substitution matrix loading
# ─────────────────────────────────────────────────────────────────────────────

def load_matrix_diagonal(name: str) -> np.ndarray:
    """
    Load a substitution matrix and extract its diagonal as float32.
    Diagonal M[aa, aa] encodes the conservation rate of amino acid aa.
    """
    from Bio.Align import substitution_matrices
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    mat = substitution_matrices.load(name)
    return np.array([mat[(aa, aa)] for aa in AA_ORDER], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# s_max computation — FIXED (MD Section 2.3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_s_max(sequences: dict,
                  M_diag: np.ndarray) -> tuple:
    """
    Compute s_max: total self-similarity weight of the longest sequence.

    FIX vs previous version:
      OLD: returned total / len(longest)   -> per-residue average
      NEW: returns total (not divided)

    Returns (s_max_total: float, longest_len: int)
    """
    longest = max(sequences.values(), key=len)
    X = encode_sequence(longest)
    total = float(sum(M_diag[aa] for aa in X if aa >= 0))
    return total, len(X)


# ─────────────────────────────────────────────────────────────────────────────
# Matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sms_matrix(sequences: dict,
                     matrix_names=None,
                     matrix_weights=None,
                     l: int = 4,
                     use_property_pass: bool = True,
                     property_weight: float = 0.3,
                     property_alphabet: str = "murphy8",
                     domains=None,
                     domain_weight: float = 2.0,
                     n_jobs: int = -1,
                     chunk_size: int = 5000,
                     out_dir: str = "output",
                     save_path=None) -> tuple:
    """
    Build the full symmetric N×N SMS similarity matrix in parallel.

    For large N (matrix RAM > _MEMMAP_THRESHOLD_BYTES), the matrix is
    backed by a np.memmap file at out_dir/checkpoints/S_matrix.dat to
    avoid OOM. Use --chunk-size to control how many pairs are computed
    per joblib batch.

    Parameters
    ----------
    sequences         : {seq_id: sequence_string}
    matrix_names      : substitution matrices to blend (default: BLOSUM45/62/80)
    matrix_weights    : blend weights (must sum to 1.0)
    l                 : minimum motif length (default 4)
    use_property_pass : enable Murphy property-group matching pass
    property_weight   : blend weight for property pass (default 0.3)
    property_alphabet : 'murphy8' or 'murphy10'
    domains           : dict[seq_id -> list of InterPro domain dicts]
    domain_weight     : multiplier for within-domain residues (default 2.0)
    n_jobs            : CPU cores for joblib (-1 = all)
    chunk_size        : pairs per joblib batch (reduce to lower peak RAM)
    out_dir           : directory for memmap file (used when N is large)
    save_path         : optional .npy path to save the matrix

    Returns
    -------
    (seq_ids: list[str], S: np.ndarray shape (N, N))
    """
    if matrix_names is None:
        matrix_names  = DEFAULT_MATRICES
    if matrix_weights is None:
        matrix_weights = DEFAULT_WEIGHTS

    assert abs(sum(matrix_weights) - 1.0) < 1e-5, (
        f"matrix_weights must sum to 1.0, got {sum(matrix_weights):.4f}"
    )
    assert len(matrix_names) == len(matrix_weights)

    ids = list(sequences.keys())
    N   = len(ids)
    pairs = list(combinations(range(N), 2))

    log.info("Building %dx%d SMS matrix  (%d pairs)", N, N, len(pairs))
    log.info("Matrices : %s", matrix_names)
    log.info("Weights  : %s", matrix_weights)
    log.info("Motif l  : %d  |  Property pass: %s (%s)  |  Domain-aware: %s",
             l, use_property_pass, property_alphabet, domains is not None)

    # Pre-compute diagonals and s_max for each matrix
    diag_list   = [load_matrix_diagonal(name) for name in matrix_names]
    smax_list   = [compute_s_max(sequences, d) for d in diag_list]
    longest_len = smax_list[0][1]

    # ── Allocate S: memmap for large N, plain ndarray for small N ─────────
    matrix_bytes = N * N * 4   # float32
    if matrix_bytes > _MEMMAP_THRESHOLD_BYTES:
        cp_dir   = os.path.join(out_dir, "checkpoints")
        os.makedirs(cp_dir, exist_ok=True)
        s_path   = os.path.join(cp_dir, "S_matrix.dat")
        log.info("Large matrix (%d MB) -- using memmap: %s",
                 matrix_bytes // (1024 * 1024), s_path)
        S = np.memmap(s_path, dtype="float32", mode="w+", shape=(N, N))
    else:
        S = np.zeros((N, N), dtype=np.float32)

    np.fill_diagonal(S, 1.0)

    def _pair(i: int, j: int) -> tuple:
        dom_i = domains.get(ids[i], []) if domains else None
        dom_j = domains.get(ids[j], []) if domains else None

        blended = 0.0
        for diag, (s_max_total, _), w in zip(diag_list, smax_list, matrix_weights):
            blended += w * compute_sms_pair(
                sequences[ids[i]],
                sequences[ids[j]],
                diag,
                longest_len,
                s_max_total,
                l                  = l,
                use_property_pass  = use_property_pass,
                property_weight    = property_weight,
                property_alphabet  = property_alphabet,
                domains1           = dom_i,
                domains2           = dom_j,
                domain_weight      = domain_weight,
            )
        return i, j, blended

    # Process in chunks to bound peak RAM for the results list
    for chunk_start in range(0, len(pairs), chunk_size):
        chunk = pairs[chunk_start: chunk_start + chunk_size]
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_pair)(i, j)
            for i, j in tqdm(
                chunk,
                desc=f"  SMS pairs (chunk {chunk_start // chunk_size + 1})",
                unit="pair",
                ncols=80,
            )
        )
        for i, j, val in results:
            S[i, j] = val
            S[j, i] = val

    if isinstance(S, np.memmap):
        S.flush()
        log.debug("memmap flushed.")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, np.asarray(S))
        log.info("Saved SMS matrix -> %s", save_path)

    return ids, np.asarray(S)


# ─────────────────────────────────────────────────────────────────────────────
# ESM-2 similarity (optional — requires fair-esm + torch)
# ─────────────────────────────────────────────────────────────────────────────

def build_esm2_matrix(sequences: dict,
                      model_name: str = "esm2_t6_8M_UR50D",
                      batch_size: int = 16) -> tuple:
    """
    Build N×N cosine similarity matrix from mean-pooled ESM-2 embeddings.

    Model choices (speed vs accuracy):
      esm2_t6_8M_UR50D    --  8M params, fast, good for N > 500 on CPU
      esm2_t12_35M_UR50D  -- 35M params, balanced
      esm2_t33_650M_UR50D -- 650M params, best accuracy, GPU recommended

    Reference: Lin et al. (2023) Science 379:1123.
    """
    try:
        import esm
        import torch
    except ImportError:
        raise ImportError(
            "ESM-2 requires: pip install fair-esm torch\n"
            "Or run without --mode esm2/hybrid."
        )

    ids = list(sequences.keys())
    N   = len(ids)

    log.info("Building ESM-2 similarity matrix (model: %s)", model_name)

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    layer_map = {
        "esm2_t6_8M_UR50D":    6,
        "esm2_t12_35M_UR50D":  12,
        "esm2_t33_650M_UR50D": 33,
        "esm2_t36_3B_UR50D":   36,
    }
    repr_layer = layer_map.get(model_name, 6)

    embeddings = np.zeros((N, model.embed_dim), dtype=np.float32)

    for start in tqdm(range(0, N, batch_size), desc="  ESM-2 batches", ncols=80):
        batch_ids  = ids[start: start + batch_size]
        batch_data = [(sid, sequences[sid]) for sid in batch_ids]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[repr_layer],
                           return_contacts=False)

        reps = results["representations"][repr_layer]  # (B, L+2, D)
        for k, sid in enumerate(batch_ids):
            seq_len = len(sequences[sid])
            emb = reps[k, 1: seq_len + 1].mean(0).cpu().numpy()
            embeddings[start + k] = emb

    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    S      = normed @ normed.T
    S      = np.clip(S, 0.0, 1.0)
    np.fill_diagonal(S, 1.0)

    log.info("ESM-2 matrix done  (device: %s)", device)
    return ids, S.astype(np.float32)


def build_hybrid_matrix(sequences: dict,
                        sms_weight: float = 0.5,
                        esm_weight: float = 0.5,
                        sms_kwargs=None,
                        esm_model: str = "esm2_t6_8M_UR50D") -> tuple:
    """
    Build hybrid similarity matrix combining SMS and ESM-2.

    S_hybrid = sms_weight * S_SMS + esm_weight * S_ESM2
    """
    assert abs(sms_weight + esm_weight - 1.0) < 1e-5, (
        "sms_weight + esm_weight must equal 1.0"
    )

    kw  = sms_kwargs or {}
    ids, S_sms  = build_sms_matrix(sequences, **kw)
    _,   S_esm2 = build_esm2_matrix(sequences, model_name=esm_model)

    S_hybrid = sms_weight * S_sms + esm_weight * S_esm2
    np.fill_diagonal(S_hybrid, 1.0)
    S_hybrid = np.clip(S_hybrid, 0.0, 1.0)

    log.info("Hybrid matrix: SMS x%.2f + ESM-2 x%.2f", sms_weight, esm_weight)
    return ids, S_hybrid.astype(np.float32)
