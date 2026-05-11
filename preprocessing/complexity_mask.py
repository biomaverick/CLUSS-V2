"""
preprocessing/complexity_mask.py
══════════════════════════════════
Mask low-complexity regions (LCRs) and optionally intrinsically
disordered regions (IDRs) before SMS computation.
Two distinct masking strategies are implemented:
1. LCR Masking — SEG-style two-threshold (Wootton & Federhen 1993)
   ─────────────────────────────────────────────────────────────────
   The ORIGINAL implementation used a single entropy threshold (2.2 bits),
   which is biologically incorrect. The true SEG algorithm uses:
     k1 = trigger threshold (default 1.8 bits): a window below k1 TRIGGERS
          a low-complexity zone.
     k2 = extend threshold (default 2.5 bits): once triggered, the zone is
          EXTENDED outward until windows exceed k2.
   Why two thresholds matter:
     - Single-threshold masking at a mid-value will either miss genuine LCRs
       (if set high) or over-mask weakly biased but functional regions (if low).
     - The trigger/extend model is consistent with how the original SEG program
       (Wootton & Federhen 1993, Comput Chem 17:149) works and correctly handles
       the bimodal entropy distribution seen in most proteins.
2. IDR Masking — IUPred3 API (Erdős et al. 2021)
   ─────────────────────────────────────────────────────────────────
   LCRs ≠ IDRs. Intrinsically disordered regions can have HIGH sequence
   entropy (varied composition) yet still produce spurious SMS matches because
   disordered regions convergently acquire similar compositions under similar
   environmental constraints (McConnell & Parker 2023, Bioinformatics).
   IUPred3 predicts disorder based on estimated residue interaction energies,
   not sequence complexity, and is the correct tool for IDR detection.
References
----------
Wootton & Federhen (1993) Comput Chem 17:149-163.
Erdős et al. (2021) Nucleic Acids Research 49:W297–W303.
McConnell & Parker (2023) Bioinformatics 39:btad732.
"""

import logging
log = logging.getLogger(__name__)


import math
import time
import os
import json
import hashlib
import requests
from collections import Counter

MASK_CHAR = "X"


# ── Shannon entropy ───────────────────────────────────────────────────────────

def shannon_entropy(window: str) -> float:
    """Shannon entropy in bits for an amino acid window string."""
    if not window:
        return 0.0
    counts = Counter(window)
    total = len(window)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
    )


# ── Two-threshold SEG masking ─────────────────────────────────────────────────

def _find_contiguous_runs(positions: set[int], n: int) -> list[tuple[int, int]]:
    """
    Find contiguous runs of integers in a set.
    Returns list of (start, end) inclusive pairs.
    """
    if not positions:
        return []
    sorted_pos = sorted(positions)
    runs: list[tuple[int, int]] = []
    run_start = sorted_pos[0]
    prev = sorted_pos[0]

    for pos in sorted_pos[1:]:
        if pos != prev + 1:
            runs.append((run_start, prev))
            run_start = pos
        prev = pos
    runs.append((run_start, prev))
    return runs


def mask_low_complexity(seq: str,
                        window: int = 12,
                        k1: float = 1.8,
                        k2: float = 2.5,
                        mask_char: str = MASK_CHAR) -> str:
    """
    Two-threshold SEG-style low-complexity masking.

    Phase 1 — Trigger:
      Slide a window of length `window` across the sequence.
      Any window with Shannon entropy < k1 adds all its positions to
      the triggered set.

    Phase 2 — Extend:
      For each contiguous triggered region, extend leftward and rightward
      as long as the extending window's entropy < k2.

    Parameters
    ----------
    seq       : amino acid sequence string (uppercase)
    window    : window size in residues (default 12)
    k1        : trigger entropy threshold in bits (default 1.8)
    k2        : extension entropy threshold in bits (default 2.5)
    mask_char : replacement character for masked positions (default 'X')

    Returns
    -------
    Masked sequence string of the same length as input.
    """
    seq_list = list(seq)
    n = len(seq_list)

    if n < window:
        # Short sequence: mask entirely if below k1
        if shannon_entropy(seq) < k1:
            return mask_char * n
        return seq

    # ── Phase 1: Trigger ─────────────────────────────────────────
    triggered: set[int] = set()
    for i in range(n - window + 1):
        w = "".join(seq_list[i: i + window])
        if shannon_entropy(w) < k1:
            for j in range(i, i + window):
                triggered.add(j)

    if not triggered:
        return seq  # nothing triggered → nothing to mask

    # ── Phase 2: Extend ──────────────────────────────────────────
    masked_positions: set[int] = set(triggered)
    runs = _find_contiguous_runs(triggered, n)

    for run_start, run_end in runs:
        # Extend left
        left = run_start
        while left > 0:
            w_start = max(0, left - window)
            w = "".join(seq_list[w_start:left])
            if len(w) == window and shannon_entropy(w) < k2:
                masked_positions.add(left - 1)
                left -= 1
            else:
                break

        # Extend right
        right = run_end
        while right < n - 1:
            w_end = min(n, right + window + 1)
            w = "".join(seq_list[right + 1: w_end])
            if len(w) == window and shannon_entropy(w) < k2:
                masked_positions.add(right + 1)
                right += 1
            else:
                break

    result = [
        mask_char if i in masked_positions else seq_list[i]
        for i in range(n)
    ]
    return "".join(result)


def mask_all_sequences(sequences: dict[str, str],
                       window: int = 12,
                       k1: float = 1.8,
                       k2: float = 2.5) -> dict[str, str]:
    """
    Apply two-threshold SEG LCR masking to all sequences.
    Logs summary statistics.
    """
    masked: dict[str, str] = {}
    total_masked = 0
    total_residues = 0

    for seq_id, seq in sequences.items():
        m = mask_low_complexity(seq, window, k1, k2)
        n_masked = m.count(MASK_CHAR)
        total_masked += n_masked
        total_residues += len(seq)
        masked[seq_id] = m

    if total_residues > 0:
        pct = 100.0 * total_masked / total_residues
        print(f"  LCR masking (SEG k1={k1}, k2={k2}): "
              f"{total_masked:,}/{total_residues:,} residues masked "
              f"({pct:.1f}%)")

    return masked


# ── IUPred3 IDR masking ───────────────────────────────────────────────────────

def _idr_cache_path(out_dir: str) -> str:
    """Return the canonical IDR cache file path for *out_dir*."""
    return os.path.join(out_dir, "idr_cache.json")


def _load_idr_cache(cache_file: str) -> dict:
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def _save_idr_cache(cache: dict, cache_file: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    with open(cache_file, "w") as fh:
        json.dump(cache, fh)


def mask_disordered_regions(seq: str,
                             threshold: float = 0.5,
                             idr_type: str = "long") -> str:
    """
    Mask intrinsically disordered residues using the IUPred3 REST API.

    IUPred3 predicts disorder based on estimated residue interaction
    energies (not sequence composition), making it complementary to LCR
    masking. IDRs with IUPred3 score > threshold are masked with 'X'.

    Parameters
    ----------
    seq       : amino acid sequence string
    threshold : IUPred3 score cutoff for disorder (default 0.5)
    idr_type  : 'long' for long disordered segments (default),
                'short' for short disordered loops

    Returns
    -------
    Masked sequence string (same length as input).

    Notes
    -----
    - Free API, no key required.
    - Rate limited: add delay between calls (handled by caller).
    - Reference: Erdős et al. (2021) Nucleic Acids Res 49:W297–W303.
    """
    url = (f"https://iupred3.elte.hu/iupred3API"
           f"?sequence={seq}&type={idr_type}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return seq  # API failure → return unmasked
        scores = r.json().get("iupred_scores", [])
        if not scores:
            return seq
        return "".join(
            MASK_CHAR if (i < len(scores) and scores[i] > threshold) else aa
            for i, aa in enumerate(seq)
        )
    except Exception:
        return seq  # Network error → return unmasked (non-fatal)


def mask_all_disordered(sequences: dict[str, str],
                         threshold: float = 0.5,
                         delay: float = 0.5,
                         out_dir: str = "output") -> dict[str, str]:
    """
    Apply IUPred3 IDR masking to all sequences.
    Results are cached to avoid re-querying.

    Parameters
    ----------
    sequences : {seq_id: sequence_string}
    threshold : IUPred3 score cutoff (default 0.5)
    delay     : seconds between API calls (default 0.5)
    out_dir   : output directory; the IDR cache is written here so it
                stays alongside all other outputs (fixes hardcoded path bug)
    """
    cache_file = _idr_cache_path(out_dir)
    cache = _load_idr_cache(cache_file)
    masked: dict[str, str] = {}
    total_masked = 0
    total_residues = 0
    fetched = 0

    for seq_id, seq in sequences.items():
        cache_key = hashlib.sha256(f"{seq}_{threshold}".encode()).hexdigest()
        if cache_key in cache:
            m = cache[cache_key]
        else:
            m = mask_disordered_regions(seq, threshold)
            cache[cache_key] = m
            fetched += 1
            time.sleep(delay)

        n_masked = m.count(MASK_CHAR)
        total_masked += n_masked
        total_residues += len(seq)
        masked[seq_id] = m

    _save_idr_cache(cache, cache_file)

    if total_residues > 0:
        pct = 100.0 * total_masked / total_residues
        print(f"  IDR masking (IUPred3 >{threshold}): "
              f"{total_masked:,}/{total_residues:,} residues masked "
              f"({pct:.1f}%)  |  {fetched} API calls made")

    return masked
