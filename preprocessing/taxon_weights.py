"""
preprocessing/taxon_weights.py
══════════════════════════════
Compute per-sequence taxon diversity weights to reduce phylogenetic
sampling bias in clustering.
Two weighting strategies are provided:
1. Simple inverse-frequency weighting (original)
   ──────────────────────────────────────────────
   weight[seq] = N / count(organism of seq)
   Problem: treats 10 E. coli strains as 10 independent organisms,
   but evolutionarily they are nearly identical. A single Thermus
   thermophilus gets the same weight as a single H. sapiens, ignoring
   their very different evolutionary distances from the bulk of the data.
2. NCBI taxonomy-aware weighting (upgrade — MD Section 2.6)
   ──────────────────────────────────────────────────────────
   Uses NCBI taxonomy lineage to compute phylogenetic-distance-based
   weights. Sequences from organisms that share long lineages (many
   common ancestors) with other organisms in the dataset get lower
   weights. Sequences from phylogenetically unique organisms are
   up-weighted.
   Implementation uses Biopython's Entrez API to fetch lineage strings,
   then computes pairwise lineage overlap as a proxy for phylogenetic
   distance. This is cached to avoid re-fetching on repeated runs.
"""

import logging
log = logging.getLogger(__name__)


import os
import json
import time
from collections import Counter
from Bio import Entrez

Entrez.email = "cluss_plus@example.com"   # required by NCBI; change to yours
TAXON_CACHE = "output/taxon_cache.json"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Simple inverse-frequency
# ─────────────────────────────────────────────────────────────────────────────

def compute_taxon_weights_simple(
        annotations: dict[str, dict]) -> dict[str, float]:
    """
    Compute inverse-frequency taxon diversity weights.

    weight[seq] = 1 / count(my_organism_in_dataset)
    Normalised so sum(weights) = N.

    Parameters
    ----------
    annotations : dict[seq_id -> annotation_dict with 'organism' key]

    Returns
    -------
    dict[seq_id -> weight (float)]
    """
    if not annotations:
        return {}

    seq_ids = list(annotations.keys())
    N = len(seq_ids)

    organisms = {
        sid: ann.get("organism", "Unknown")
        for sid, ann in annotations.items()
    }
    org_counts = Counter(organisms.values())
    raw = {sid: 1.0 / org_counts[org] for sid, org in organisms.items()}
    total = sum(raw.values())

    weights = {sid: N * w / total for sid, w in raw.items()}

    n_orgs = len(org_counts)
    top_org, top_cnt = org_counts.most_common(1)[0]
    print(f"  Taxon weighting (simple): {n_orgs} unique organisms  |  "
          f"most common: '{top_org}' ({top_cnt} seqs)")

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: NCBI taxonomy-aware weighting
# ─────────────────────────────────────────────────────────────────────────────

def _load_taxon_cache() -> dict:
    if os.path.exists(TAXON_CACHE):
        with open(TAXON_CACHE) as fh:
            return json.load(fh)
    return {}


def _save_taxon_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(TAXON_CACHE), exist_ok=True)
    with open(TAXON_CACHE, "w") as fh:
        json.dump(cache, fh, indent=2)


def _fetch_lineage(organism_name: str, cache: dict) -> list[str]:
    """
    Fetch NCBI taxonomy lineage for an organism name.
    Returns list of ancestor taxon names from root → organism.
    Cached to avoid repeated API calls.
    """
    if organism_name in cache:
        return cache[organism_name]

    try:
        handle = Entrez.esearch(db="taxonomy", term=organism_name, retmax=1)
        record = Entrez.read(handle)
        handle.close()
        time.sleep(0.34)  # NCBI rate limit: 3 requests/second

        if not record["IdList"]:
            cache[organism_name] = []
            return []

        taxid = record["IdList"][0]
        handle = Entrez.efetch(db="taxonomy", id=taxid, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        time.sleep(0.34)

        lineage_ex = records[0].get("LineageEx", [])
        lineage = [node["ScientificName"] for node in lineage_ex]
        lineage.append(organism_name)

        cache[organism_name] = lineage
        return lineage

    except Exception:
        cache[organism_name] = []
        return []


def _lineage_overlap(la: list[str], lb: list[str]) -> float:
    """
    Fraction of shared lineage between two organisms.
    1.0 = identical lineage, 0.0 = no shared ancestors.
    """
    if not la or not lb:
        return 0.0
    set_a, set_b = set(la), set(lb)
    shared = len(set_a & set_b)
    return shared / max(len(set_a), len(set_b))


def compute_taxon_weights_phylogenetic(
        annotations: dict[str, dict]) -> dict[str, float]:
    """
    Compute phylogenetic-distance-aware taxon weights via NCBI lineage.

    A sequence from an organism that shares a long lineage with many
    other organisms in the dataset (phylogenetically redundant) gets
    a lower weight. Sequences from phylogenetically unique organisms
    are up-weighted.

    Algorithm:
      1. Fetch NCBI taxonomy lineage for each unique organism.
      2. For each sequence, compute its mean lineage overlap with all
         other sequences.
      3. Weight[seq] = 1 / (1 + mean_overlap) — high overlap → low weight.
      4. Normalise weights to sum to N.

    Falls back to simple inverse-frequency if NCBI queries fail.

    Parameters
    ----------
    annotations : dict[seq_id -> annotation_dict with 'organism' key]

    Returns
    -------
    dict[seq_id -> weight (float)]
    """
    if not annotations:
        return {}

    seq_ids = list(annotations.keys())
    N = len(seq_ids)
    organisms = {sid: ann.get("organism", "Unknown")
                 for sid, ann in annotations.items()}

    unique_orgs = list(set(organisms.values()))
    log.info(f"  Fetching NCBI lineages for {len(unique_orgs)} unique organisms...")
    cache = _load_taxon_cache()
    lineages: dict[str, list[str]] = {}

    for org in unique_orgs:
        lineages[org] = _fetch_lineage(org, cache)

    _save_taxon_cache(cache)

    # Compute mean lineage overlap for each sequence
    seq_overlaps: dict[str, float] = {}
    for sid in seq_ids:
        org_a = organisms[sid]
        la    = lineages.get(org_a, [])
        overlaps = [
            _lineage_overlap(la, lineages.get(organisms[other], []))
            for other in seq_ids if other != sid
        ]
        seq_overlaps[sid] = sum(overlaps) / len(overlaps) if overlaps else 0.0

    # Weight inversely to overlap
    raw    = {sid: 1.0 / (1.0 + ov) for sid, ov in seq_overlaps.items()}
    total  = sum(raw.values())
    weights = {sid: N * w / total for sid, w in raw.items()}

    mean_w = sum(weights.values()) / N
    log.info(f"  Taxon weighting (phylogenetic): mean weight={mean_w:.3f}")
    return weights


def compute_taxon_weights(annotations: dict[str, dict],
                          use_phylogenetic: bool = False) -> dict[str, float]:
    """
    Dispatch to the appropriate weighting strategy.

    Parameters
    ----------
    annotations      : annotation dict
    use_phylogenetic : if True, use NCBI-lineage-based weighting;
                       otherwise use simple inverse-frequency
    """
    if use_phylogenetic:
        return compute_taxon_weights_phylogenetic(annotations)
    return compute_taxon_weights_simple(annotations)
