"""
evaluation/q_measure.py
══════════════════════════
Clustering quality evaluation.

Metrics implemented (MD Section 2.7)
──────────────────────────────────────
1. Q-measure (Kelil et al. 2007 — paper-specific metric)
   Q = (Σ P_i - U) / N × 100  clamped to [0, 100]
   FIX: added clamp — Q went negative when U was large.

2. Adjusted Rand Index (ARI) — standard external metric
   Measures agreement between two clusterings, corrected for chance.
   Range: [-1, 1] where 1 = perfect agreement, 0 = random.

3. Normalised Mutual Information (NMI) — standard external metric
   Information-theoretic agreement between clusterings.
   Range: [0, 1] where 1 = perfect.

4. Silhouette Score — internal metric (no reference needed)
   Measures how similar a sequence is to its own cluster vs. others.
   Range: [-1, 1] where 1 = well separated clusters.
   Uses the precomputed distance matrix (D = 1 - S).
"""

import os
import json
import numpy as np
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score,
                             silhouette_score)


def load_reference(file_path: str) -> dict[str, str]:
    """
    Load reference classification from a two-column TSV file.

    Format (tab-separated, optional header starting with #):
      seq_id <TAB> functional_group

    Parameters
    ----------
    file_path : path to the reference TSV file

    Returns
    -------
    dict[seq_id -> functional_group_string]
    """
    ref: dict[str, str] = {}
    with open(file_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ref[parts[0].strip()] = parts[1].strip()

    print(f"  Reference: {len(ref):,} labelled entries from {file_path}")
    return ref


def _build_label_arrays(seq_ids: list[str],
                        clusters: list[list[str]],
                        orphans: list[str],
                        reference: dict[str, str]
                        ) -> tuple[list, list]:
    """
    Build parallel true-label and predicted-label arrays over seq_ids.
    Orphans receive the predicted label 'orphan'.
    Sequences not in reference receive true label 'unknown'.
    """
    cluster_ids: dict[str, str] = {}
    for i, c in enumerate(clusters):
        for sid in c:
            cluster_ids[sid] = str(i)
    for sid in orphans:
        cluster_ids[sid] = "orphan"

    true_labels = [reference.get(sid, "unknown") for sid in seq_ids]
    pred_labels = [cluster_ids.get(sid, "orphan") for sid in seq_ids]
    return true_labels, pred_labels


def compute_q_measure(clusters: list[list[str]],
                      orphans: list[str],
                      reference: dict[str, str],
                      taxon_weights: dict[str, float] | None = None) -> float:
    """
    Compute Q-measure as defined in Kelil et al. (2007).

        Q = max(0, (Σ P_i - U) / N) × 100

    Where:
      N   = total sequences (clustered + orphans)
      P_i = count (or weighted count) of the majority functional group
            in cluster i according to the reference classification
      U   = number of orphan sequences

    Parameters
    ----------
    clusters      : list of lists of seq_ids
    orphans       : list of unclustered seq_ids
    reference     : dict[seq_id -> functional_group]
    taxon_weights : optional per-sequence weights for weighted P_i

    Returns
    -------
    Q-measure in [0.0, 100.0]
    """
    N = sum(len(c) for c in clusters) + len(orphans)
    if N == 0:
        return 0.0

    U       = len(orphans)
    total_P = 0.0

    for cluster in clusters:
        group_scores: dict[str, float] = {}
        for sid in cluster:
            if sid not in reference:
                continue
            group = reference[sid]
            w     = (taxon_weights.get(sid, 1.0)
                     if taxon_weights else 1.0)
            group_scores[group] = group_scores.get(group, 0.0) + w
        if group_scores:
            total_P += max(group_scores.values())

    # FIXED: clamp to [0, 100]
    return max(0.0, min(100.0, (total_P - U) / N * 100.0))


def compute_standard_metrics(clusters: list[list[str]],
                             orphans: list[str],
                             reference: dict[str, str],
                             S: np.ndarray,
                             seq_ids: list[str]) -> dict[str, float]:
    """
    Compute ARI, NMI, and Silhouette score (MD Section 2.7).

    Parameters
    ----------
    clusters : list of lists of seq_ids
    orphans  : list of unclustered seq_ids
    reference: dict[seq_id -> functional_group]
    S        : N×N similarity matrix (used for Silhouette via D = 1 - S)
    seq_ids  : ordered list of all sequence IDs (matches S rows/cols)

    Returns
    -------
    dict with keys: ARI, NMI, Silhouette
    """
    true_labels, pred_labels = _build_label_arrays(
        seq_ids, clusters, orphans, reference
    )

    ari = float(adjusted_rand_score(true_labels, pred_labels))
    nmi = float(normalized_mutual_info_score(true_labels, pred_labels))

    # Silhouette requires ≥ 2 clusters and ≥ 2 samples per cluster
    sil = None
    unique_pred = set(pred_labels)
    if len(unique_pred) >= 2 and len(seq_ids) >= 4:
        try:
            D   = 1.0 - S                         # distance matrix
            D   = np.clip(D, 0.0, None)
            sil = float(silhouette_score(D, pred_labels, metric="precomputed"))
        except Exception:
            pass

    result: dict[str, float] = {"ARI": round(ari, 4), "NMI": round(nmi, 4)}
    if sil is not None:
        result["Silhouette"] = round(sil, 4)

    return result


def save_metrics(Q: float,
                 clusters: list,
                 orphans: list,
                 runtime_seconds: float,
                 output_dir: str = "output",
                 standard_metrics: dict | None = None) -> None:
    """
    Save Q-measure and run statistics to metrics.json.

    Parameters
    ----------
    Q                : Q-measure value
    clusters         : extracted clusters
    orphans          : orphan sequences
    runtime_seconds  : total runtime
    output_dir       : output directory
    standard_metrics : optional dict from compute_standard_metrics()
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics: dict = {
        "Q_measure":         round(Q, 4),
        "n_clusters":        len(clusters),
        "n_orphans":         len(orphans),
        "n_total":           sum(len(c) for c in clusters) + len(orphans),
        "cluster_sizes":     sorted([len(c) for c in clusters], reverse=True),
        "runtime_seconds":   round(runtime_seconds, 2),
    }

    if standard_metrics:
        metrics.update(standard_metrics)

    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"  Saved metrics → {path}")

    # Print summary
    print(f"\n  ── Evaluation Summary ──────────────────────")
    print(f"  Q-measure  : {Q:.2f}%")
    if standard_metrics:
        for key, val in standard_metrics.items():
            print(f"  {key:<12}: {val:.4f}")
    print(f"  ────────────────────────────────────────────")
