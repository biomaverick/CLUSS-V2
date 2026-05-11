"""
evaluation/go_enrichment.py
════════════════════════════
GO term enrichment analysis per cluster with semantic reduction.
Two steps
──────────
Step 1 — Statistical enrichment (Fisher's exact test + BH FDR)
  For each cluster, test which GO terms are over-represented compared
  to the full dataset. Uses scipy.stats.fisher_exact and
  statsmodels.stats.multitest for Benjamini-Hochberg FDR correction.
Step 2 — Semantic GO term reduction (MD Section 3.5)
  Without semantic reduction, enriched GO term lists contain many
  parent-child redundancies (e.g. both 'protein phosphorylation'
  GO:0006468 and its parent 'phosphorylation' GO:0016310 may both
  be significant). This clutters the output and misleads interpretation.
  Requires: pip install goatools
  GO OBO file: downloaded automatically to output/go-basic.obo if absent.
References
──────────
Ashburner et al. (2000) Nat Genet 25:25–29. (Gene Ontology)
Klopfenstein et al. (2018) Sci Rep 8:10872. (goatools)
"""

import os
import urllib.request
from collections import Counter
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

OBO_URL  = "http://purl.obolibrary.org/obo/go/go-basic.obo"
OBO_PATH = "output/go-basic.obo"


# ─────────────────────────────────────────────────────────────────────────────
# GO OBO file management
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_obo() -> str:
    """
    Download go-basic.obo if it doesn't exist locally.
    Returns the local file path.
    """
    if not os.path.exists(OBO_PATH):
        os.makedirs(os.path.dirname(OBO_PATH), exist_ok=True)
        print(f"  Downloading GO OBO file → {OBO_PATH} ...")
        urllib.request.urlretrieve(OBO_URL, OBO_PATH)
        print("  GO OBO downloaded.")
    return OBO_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Semantic reduction
# ─────────────────────────────────────────────────────────────────────────────

def reduce_go_terms(go_list: list[str]) -> list[str]:
    """
    Remove semantically redundant GO terms using the GO DAG.

    A term T is removed if any other term in go_list is a descendant
    of T (i.e., T is an ancestor and therefore less specific).
    Only the most specific (deepest) terms are retained.

    Parameters
    ----------
    go_list : list of GO accession strings (e.g. ['GO:0006468', ...])

    Returns
    -------
    Non-redundant list of GO terms (most specific terms only).

    Falls back to returning go_list unchanged if goatools is unavailable.
    """
    if not go_list:
        return go_list

    try:
        from goatools.obo_parser import GODag
    except ImportError:
        return go_list   # goatools not installed — skip reduction

    try:
        obo_path = _ensure_obo()
        dag = GODag(obo_path, optional_attrs={"relationship"}, load_obsolete=False)
    except Exception:
        return go_list   # OBO parse failed — skip reduction

    go_set = set(go_list)

    def _is_ancestor_of_any(term_id: str) -> bool:
        """True if term_id is an ancestor of any other term in go_set."""
        if term_id not in dag:
            return False
        for other_id in go_set:
            if other_id == term_id or other_id not in dag:
                continue
            other_parents = dag[other_id].get_all_parents()
            if term_id in other_parents:
                return True
        return False

    non_redundant = [t for t in go_list if not _is_ancestor_of_any(t)]
    return non_redundant if non_redundant else go_list


# ─────────────────────────────────────────────────────────────────────────────
# Enrichment analysis
# ─────────────────────────────────────────────────────────────────────────────

def go_enrichment(clusters: list[list[str]],
                  annotations: dict[str, dict],
                  alpha: float = 0.05,
                  semantic_reduction: bool = True) -> dict[int, list[str]]:
    """
    Fisher's exact test for GO term enrichment per cluster,
    with Benjamini-Hochberg FDR correction and optional semantic reduction.

    Parameters
    ----------
    clusters           : list of lists of seq_ids (one list per cluster)
    annotations        : dict[seq_id -> annotation_dict with 'go_terms' key]
    alpha              : FDR significance threshold (default 0.05)
    semantic_reduction : remove ancestor GO terms from enriched sets

    Returns
    -------
    dict[cluster_index -> list of enriched (non-redundant) GO term strings]
    """
    # Background: all GO terms across the full dataset
    all_go: list[str] = []
    annotated_N = 0

    for ann in annotations.values():
        terms = ann.get("go_terms", [])
        if terms:
            annotated_N += 1
        all_go.extend(terms)

    background_go = Counter(all_go)
    N_total       = annotated_N

    if N_total == 0 or not background_go:
        print("  GO enrichment: no GO annotations available — skipping")
        return {}

    enriched_per_cluster: dict[int, list[str]] = {}

    for idx, cluster in enumerate(clusters):
        cluster_go: list[str] = []
        N_cluster = 0

        for sid in cluster:
            terms = annotations.get(sid, {}).get("go_terms", [])
            if terms:
                N_cluster += 1
            cluster_go.extend(terms)

        if N_cluster == 0:
            enriched_per_cluster[idx] = []
            continue

        cluster_counts = Counter(cluster_go)
        terms_tested:  list[str]   = []
        pvals:         list[float] = []

        for term, k in cluster_counts.items():
            K   = background_go[term]
            n   = N_cluster
            N   = N_total
            nwk = max(0, N - n - (K - k))   # not-in-cluster without term

            table = [[k,      n - k],
                     [K - k,  nwk]]

            _, p = fisher_exact(table, alternative="greater")
            terms_tested.append(term)
            pvals.append(p)

        if not pvals:
            enriched_per_cluster[idx] = []
            continue

        reject, _, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        enriched = [t for t, r in zip(terms_tested, reject) if r]

        # Semantic reduction — remove ancestor terms (MD Section 3.5)
        if semantic_reduction and enriched:
            enriched = reduce_go_terms(enriched)

        enriched_per_cluster[idx] = enriched

    n_enriched = sum(1 for v in enriched_per_cluster.values() if v)
    print(f"  GO enrichment: {n_enriched}/{len(clusters)} clusters have "
          f"enriched terms (FDR < {alpha}"
          f"{', semantic reduction applied' if semantic_reduction else ''})")

    return enriched_per_cluster
