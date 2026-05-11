"""
similarity/sms_engine.py
══════════════════════════
Core SMS (Substitution Matching Similarity) engine.
Implements the algorithm from Kelil et al. (2007) BMC Bioinformatics 8:286,
with all bug fixes and biological upgrades applied.
Biological upgrades applied
────────────────────────────────────────────────────
  1. Property-group alphabet replaced with Murphy et al. (2000) empirically
     derived reduced alphabets (Murphy8 and Murphy10), replacing the
     biologically incorrect Taylor 6-class scheme that grouped Cys with
     Ser/Thr (biochemically invalid — Cys forms disulfide bonds, coordinates
     metals, and has unique redox chemistry that makes it non-exchangeable
     with Ser or Thr in any biologically meaningful sense).
  2. Three alphabet choices exposed via `property_alphabet` parameter:
       'murphy8'  — 8-class (used in the property-group pass by default)
       'murphy10' — 10-class (retains 93% of BLOSUM62 information)
  3. Domain-aware SMS scoring: when InterPro domain boundaries are available,
     residues inside known globular domains are up-weighted, linker regions
     are down-weighted. This focuses SMS on biologically meaningful segments.
References
──────────
Murphy et al. (2000) Protein Engineering 13(3):149–152.
Kelil et al. (2007) BMC Bioinformatics 8:286.
"""

import logging
log = logging.getLogger(__name__)


import numpy as np
from numba import njit


# ─────────────────────────────────────────────────────────────────────────────
# Amino acid encoding
# ─────────────────────────────────────────────────────────────────────────────

AA_ORDER  = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT: dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}
MASK_INT  = -1   # masked positions (LCR or IDR) — skipped in all matching


def encode_sequence(seq: str) -> np.ndarray:
    """
    Encode an amino acid string to a numpy int32 array.
    Masked positions ('X') and unknown characters → MASK_INT (-1).
    """
    arr = np.empty(len(seq), dtype=np.int32)
    for idx, aa in enumerate(seq):
        arr[idx] = AA_TO_INT.get(aa, MASK_INT)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Murphy reduced alphabets (MD Sections 2.1 and 3.4)
# ─────────────────────────────────────────────────────────────────────────────
# Both derived from hierarchical clustering of BLOSUM62 substitution
# frequencies (Murphy et al. 2000 Protein Engineering 13:149–152).
# Cysteine (C) is no longer grouped with
# Ser/Thr. Cys is unique because:
#   - Disulfide bond formation (structural / catalytic)
#   - Metal coordination (Zn-finger, Fe-S clusters)
#   - Redox sensor function
#   - BLOSUM62 Cys row is almost entirely off-diagonal
# ─────────────────────────────────────────────────────────────────────────────
# Murphy 8-class (default for property pass)
_MURPHY8: dict[str, int] = {
    "I": 0, "L": 0, "V": 0, "M": 0,    # hydrophobic/aliphatic
    "F": 1, "Y": 1, "W": 1,             # aromatic
    "S": 2, "T": 2, "N": 2, "Q": 2,    # polar uncharged (true polar)
    "K": 3, "R": 3, "H": 3,            # positively charged
    "D": 4, "E": 4,                     # negatively charged
    "C": 5,                             # cysteine — unique class
    "G": 6, "A": 6,                     # small/glycine
    "P": 7,                             # proline — helix breaker
}

# Murphy 10-class (retains 93% of BLOSUM62 information content)
# Corrected: Met belongs with aliphatic hydrophobics (I, L, V), not with Cys.
# Cys is a singleton class in Murphy10 for the same biochemical reasons as in
# Murphy8 — disulfide bonds, metal coordination, unique redox chemistry.
# See Murphy et al. (2000) Protein Engineering 13(3):149-152, Table 1.
_MURPHY10: dict[str, int] = {
    "I": 0, "L": 0, "V": 0, "M": 0,    # aliphatic hydrophobic (corrected)
    "C": 1,                              # cysteine — unique singleton (corrected)
    "F": 2, "Y": 2, "W": 2,            # aromatic
    "H": 3,                             # imidazole
    "K": 4, "R": 4,                     # positive charge
    "D": 5, "E": 5,                     # negative charge
    "N": 6, "Q": 6,                     # amide
    "S": 7, "T": 7,                     # hydroxyl
    "A": 8, "G": 8,                     # small/glycine
    "P": 9,                             # proline
}

ALPHABETS: dict[str, dict[str, int]] = {
    "murphy8":  _MURPHY8,
    "murphy10": _MURPHY10,
}


def encode_property_groups(seq: str,
                           alphabet: str = "murphy8") -> np.ndarray:
    prop_map = ALPHABETS.get(alphabet, _MURPHY8)
    arr = np.empty(len(seq), dtype=np.int32)
    for idx, aa in enumerate(seq):
        if aa == "X" or aa not in prop_map:
            arr[idx] = MASK_INT
        else:
            arr[idx] = prop_map[aa]
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Domain-aware position weight mask (MD Section 3.6)
# ─────────────────────────────────────────────────────────────────────────────

def make_domain_mask(seq_len: int,
                     domains: list[dict],
                     domain_weight: float = 2.0,
                     linker_weight: float = 0.5) -> np.ndarray:
    """
    Build a per-position weight array for domain-aware SMS scoring.

    Residues inside annotated InterPro domain boundaries receive
    `domain_weight`; linker regions receive `linker_weight`.

    Parameters
    ----------
    seq_len       : length of the sequence
    domains       : list of {'start': int, 'end': int} dicts (1-indexed)
    domain_weight : multiplier for residues inside domains (default 2.0)
    linker_weight : multiplier for residues outside domains (default 0.5)

    Returns
    -------
    np.ndarray shape (seq_len,) dtype float32
    """
    mask = np.full(seq_len, linker_weight, dtype=np.float32)
    for dom in domains:
        s = int(dom.get("start", 1)) - 1    # convert to 0-indexed
        e = int(dom.get("end",   1))        # end is inclusive (1-indexed)
        s = max(0, s)
        e = min(seq_len, e)
        mask[s:e] = domain_weight
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Seed detection (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

@njit
def find_seeds(X: np.ndarray, Y: np.ndarray, l: int) -> list:
    """
    Find all diagonal runs of exactly length l as seeds for expansion.

    Iterates all diagonals of the implicit comparison matrix of X and Y.
    A seed is a consecutive run of identical non-masked characters of
    exactly length l, recorded at its START positions in X and Y.
    Parameters
    ----------
    X, Y : int32 arrays (-1 = masked, skip)
    l    : minimum match length

    Returns list of (pos_x, pos_y, length=l) tuples.
    """
    seeds = []
    len_x = len(X)
    len_y = len(Y)

    for offset in range(-(len_y - 1), len_x):
        i = max(0, offset)
        j = max(0, -offset)

        run_len   = 0
        run_start = i   # start of current run in X

        while i < len_x and j < len_y:
            if X[i] != MASK_INT and Y[j] != MASK_INT and X[i] == Y[j]:
                if run_len == 0:
                    run_start = i          # record true start in X
                run_len += 1

                if run_len == l:
                    # FIXED: run_start is already the correct X start position
                    seeds.append((run_start, j - (l - 1), l))
                    # Do NOT reset — expansion handles longer-than-l runs

            else:
                run_len = 0

            i += 1
            j += 1

    return seeds
# ─────────────────────────────────────────────────────────────────────────────
# Seed expansion
# ─────────────────────────────────────────────────────────────────────────────

@njit
def expand_seed(X: np.ndarray, Y: np.ndarray,
                px: int, py: int, length: int):
    """
    Expand a seed match maximally in both directions along its diagonal.
    Returns (new_px, new_py, new_length).
    """
    x = px
    y = py
    le = length

    # Expand left
    while (x > 0 and y > 0
           and X[x - 1] != MASK_INT and Y[y - 1] != MASK_INT
           and X[x - 1] == Y[y - 1]):
        x  -= 1
        y  -= 1
        le += 1

    # Expand right
    ex = x + le
    ey = y + le
    while (ex < len(X) and ey < len(Y)
           and X[ex] != MASK_INT and Y[ey] != MASK_INT
           and X[ex] == Y[ey]):
        ex += 1
        ey += 1
        le += 1

    return x, y, le


# ─────────────────────────────────────────────────────────────────────────────
# Maximality filter
# ─────────────────────────────────────────────────────────────────────────────

def maximal_filter(expanded: list) -> list:
    """
    Retain only maximal matched subsequences.
    A match (px, py, l) is non-maximal if it is fully contained within
    another match both by X-position and Y-position. Deduplication and
    longest-first sorting ensures correctness in O(n²) with small n.
    """
    expanded = list(dict.fromkeys(expanded))   # fast dedup
    expanded.sort(key=lambda t: (-t[2], t[0], t[1]))

    E: list = []
    for cand in expanded:
        px, py, l = cand
        contained = False
        for ex_px, ex_py, ex_l in E:
            if (ex_px <= px
                    and ex_py <= py
                    and ex_px + ex_l >= px + l
                    and ex_py + ex_l >= py + l):
                contained = True
                break
        if not contained:
            E.append(cand)

    return E


# ─────────────────────────────────────────────────────────────────────────────
# Internal scoring function
# ─────────────────────────────────────────────────────────────────────────────

def _sms_score_raw(X: np.ndarray, Y: np.ndarray,
                   M_diag: np.ndarray,
                   domain_mask1: np.ndarray | None,
                   domain_mask2: np.ndarray | None,
                   l: int) -> float:
    """
    Compute raw weighted match score for encoded arrays X and Y.
    Uses a covered[] array to prevent double-counting overlapping matches.
    Optionally applies domain-position weight masks.
    Returns total weight (not normalised).
    """
    seeds = find_seeds(X, Y, l)
    if not seeds:
        return 0.0

    expanded = [expand_seed(X, Y, px, py, ln) for px, py, ln in seeds]
    E        = maximal_filter(expanded)

    covered      = np.zeros(len(X), dtype=np.bool_)
    total_weight = 0.0

    for px, py, length in E:
        for k in range(length):
            pos_x = px + k
            if 0 <= pos_x < len(X) and not covered[pos_x]:
                aa_int = X[pos_x]
                if aa_int != MASK_INT and 0 <= aa_int < len(M_diag):
                    base  = float(M_diag[aa_int])
                    # Domain weight for X position (default 1.0 if no mask)
                    dw1 = float(domain_mask1[pos_x]) if domain_mask1 is not None else 1.0
                    # Domain weight for Y position
                    pos_y = py + k
                    dw2 = float(domain_mask2[pos_y]) if (
                        domain_mask2 is not None and 0 <= pos_y < len(domain_mask2)
                    ) else 1.0
                    # Use geometric mean of the two domain weights
                    domain_factor = (dw1 * dw2) ** 0.5
                    total_weight += base * domain_factor
                covered[pos_x] = True

    return total_weight


# ─────────────────────────────────────────────────────────────────────────────
# Public SMS interface
# ─────────────────────────────────────────────────────────────────────────────

def compute_sms_pair(seq1: str,
                     seq2: str,
                     M_diag: np.ndarray,
                     longest_len: int,
                     s_max_total: float,
                     l: int = 4,
                     use_property_pass: bool = True,
                     property_weight: float = 0.3,
                     property_alphabet: str = "murphy8",
                     domains1: list[dict] | None = None,
                     domains2: list[dict] | None = None,
                     domain_weight: float = 2.0) -> float:
    """
    Compute SMS similarity between two protein sequences.
    Normalisation (MD Section 2.3, corrected):
      S = combined_raw / s_max_total
    s_max_total is the total diagonal weight of the longest sequence.
    Dividing directly gives S in [0,1] for all pair lengths.
    Parameters
    ----------
    seq1, seq2          : amino acid strings (may contain 'X' for masked)
    M_diag              : substitution matrix diagonal, shape (20,)
    longest_len         : length of the longest sequence in the family
    s_max_total         : total self-similarity weight of the longest sequence
    l                   : minimum motif length (default 4)
    use_property_pass   : enable Murphy property-group matching pass
    property_weight     : blend weight for property pass (0–1, default 0.3)
    property_alphabet   : 'murphy8' or 'murphy10'
    domains1, domains2  : InterPro domain dicts for seq1 and seq2
    domain_weight       : multiplier for within-domain residues (default 2.0)
    Returns
    -------
    float in [0, 1]
    """
    X = encode_sequence(seq1)
    Y = encode_sequence(seq2)

    if len(X) == 0 or len(Y) == 0 or s_max_total <= 0 or longest_len <= 0:
        return 0.0

    # Build domain masks (None if no domain info available)
    dm1 = make_domain_mask(len(X), domains1, domain_weight) if domains1 else None
    dm2 = make_domain_mask(len(Y), domains2, domain_weight) if domains2 else None

    # Exact-match SMS pass
    exact_raw = _sms_score_raw(X, Y, M_diag, dm1, dm2, l)

    if use_property_pass:
        Xp = encode_property_groups(seq1, property_alphabet)
        Yp = encode_property_groups(seq2, property_alphabet)
        prop_raw = _sms_score_raw(Xp, Yp, M_diag, dm1, dm2, l)
        # Blend: exact dominates, property adds sensitivity for conservative subs
        combined_raw = ((1.0 - property_weight) * exact_raw
                        + property_weight       * prop_raw)
    else:
        combined_raw = exact_raw

    # Normalise: S = combined_raw / s_max_total.
    # s_max_total is the total diagonal weight of the longest sequence.
    # Dividing combined_raw directly by s_max_total gives S in [0,1] for all
    # pairs, including pairs of short sequences — no ceiling artefact.
    # (Previously the code divided by longest_len AND then by s_max_total,
    # which reduced the self-similarity of the longest sequence to 1/longest_len.)
    return min(1.0, combined_raw / s_max_total)


# ---------------------------------------------------------------------------
# 11 — Numba JIT warm-up
# ---------------------------------------------------------------------------
# @njit-decorated functions (find_seeds, expand_seed) compile on the FIRST
# call, adding 10-30 s of invisible delay that looks like a hang on large
# datasets.  Trigger compilation now with a tiny dummy input so the progress
# bar starts promptly on real data.
import numpy as _np_warmup
_dummy = _np_warmup.array([0, 1, 2, 3], dtype=_np_warmup.int32)
try:
    find_seeds(_dummy, _dummy, 2)
except Exception:
    pass  # never raises on valid input; guard against future signature changes
del _np_warmup, _dummy
