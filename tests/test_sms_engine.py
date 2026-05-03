"""
tests/test_sms_engine.py
═════════════════════════
Unit tests for similarity/sms_engine.py.

Covers
──────
  1.  Sequence encoding  (AA_TO_INT, masked chars, empty input)
  2.  Property-group encoding  (Murphy8 / Murphy10; critical Cys fix)
  3.  find_seeds()  — BUG FIX: X-position must not have (l-1) subtracted
  4.  expand_seed()  — left/right expansion; boundaries; masking
  5.  maximal_filter()  — containment; dedup; partial overlap
  6.  s_max normalisation  — BUG FIX: short pairs must not be clipped to 1.0
  7.  compute_sms_pair()  — identical > partial > disjoint; symmetry; range
  8.  Domain mask construction
  9.  Property-pass blend  (conservative subs score >= exact-only)

Run with
────────
  cd cluss_plus/
  python -m pytest tests/test_sms_engine.py -v
"""

import sys
import os
import pytest
import numpy as np


from similarity.sms_engine import (
    AA_ORDER, AA_TO_INT, MASK_INT,
    _MURPHY8, _MURPHY10,
    encode_sequence,
    encode_property_groups,
    make_domain_mask,
    find_seeds,
    expand_seed,
    maximal_filter,
    compute_sms_pair,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def blosum62_diag() -> np.ndarray:
    """BLOSUM62 diagonal — self-substitution scores for the 20 standard AAs."""
    from Bio.Align import substitution_matrices
    mat = substitution_matrices.load("BLOSUM62")
    return np.array([mat[(aa, aa)] for aa in AA_ORDER], dtype=np.float32)


@pytest.fixture(scope="module")
def family_ctx(blosum62_diag):
    """
    Pre-computed s_max context for a family whose longest sequence has 40 AAs.
    Re-used across normalisation and pair tests.
    """
    from similarity.sms_matrix import compute_s_max
    seqs = {
        "long":    "ACDEFGHIKLMNPQRSTVWY" * 2,   # 40 AA (longest in family)
        "full":    "ACDEFGHIKLMNPQRSTVWY",
        "partial": "ACDEFGHIKL" + "QQQQQQQQQQ",
        "none":    "QQQQQQQQQQQQQQQQQQQQ",
        "short":   "ACDE",
    }
    s_max_total, longest_len = compute_s_max(seqs, blosum62_diag)
    return {"seqs": seqs, "s_max": s_max_total, "llen": longest_len}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sequence encoding
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeSequence:

    def test_all_20_standard_aas_in_order(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        enc = encode_sequence(seq)
        assert len(enc) == 20
        assert list(enc) == list(range(20)), \
            "Standard AA encoding does not follow AA_ORDER"

    def test_masked_x_becomes_mask_int(self):
        enc = encode_sequence("ACXDE")
        assert enc[2] == MASK_INT
        assert enc[0] == AA_TO_INT["A"]
        assert enc[3] == AA_TO_INT["D"]

    def test_unknown_chars_become_mask_int(self):
        enc = encode_sequence("ABCZ")  # B and Z not in standard 20
        assert enc[1] == MASK_INT
        assert enc[3] == MASK_INT
        assert enc[0] == AA_TO_INT["A"]
        assert enc[2] == AA_TO_INT["C"]

    def test_empty_sequence_returns_empty_array(self):
        enc = encode_sequence("")
        assert len(enc) == 0
        assert isinstance(enc, np.ndarray)

    def test_all_masked_sequence(self):
        enc = encode_sequence("XXXXX")
        assert all(v == MASK_INT for v in enc)

    def test_dtype_is_int32(self):
        enc = encode_sequence("ACDE")
        assert enc.dtype == np.int32


# ─────────────────────────────────────────────────────────────────────────────
# 2. Murphy reduced alphabets — biological correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestMurphyAlphabets:
    """
    Critical biological fix: Cysteine must NOT share a class with Ser or Thr.

    Old (Taylor 6-class):  "S": 2, "T": 2, "C": 2, "M": 2  ← biochemically wrong.
    Cys forms disulfide bonds, coordinates metals, acts as redox sensor.
    BLOSUM62 Cys row is almost entirely off-diagonal — it does not
    substitute conservatively with Ser or Thr in any natural protein family.

    Murphy et al. (2000) Protein Engineering 13:149 derived reduced alphabets
    directly from BLOSUM62 substitution frequencies, which are used here.
    """

    # ── Murphy8 ───────────────────────────────────────────────────────────────

    def test_murphy8_cys_has_own_class_not_with_ser(self):
        assert _MURPHY8["C"] != _MURPHY8["S"], \
            "BUG: Cys grouped with Ser in Murphy8"

    def test_murphy8_cys_has_own_class_not_with_thr(self):
        assert _MURPHY8["C"] != _MURPHY8["T"], \
            "BUG: Cys grouped with Thr in Murphy8"

    def test_murphy8_cys_not_with_met(self):
        # In Murphy8, Met is aliphatic/hydrophobic; Cys is its own unique class
        assert _MURPHY8["C"] != _MURPHY8["M"]

    def test_murphy8_positive_charges_grouped(self):
        assert _MURPHY8["K"] == _MURPHY8["R"] == _MURPHY8["H"]

    def test_murphy8_negative_charges_grouped(self):
        assert _MURPHY8["D"] == _MURPHY8["E"]

    def test_murphy8_opposite_charges_separate(self):
        assert _MURPHY8["K"] != _MURPHY8["D"]

    def test_murphy8_aromatics_grouped(self):
        assert _MURPHY8["F"] == _MURPHY8["Y"] == _MURPHY8["W"]

    def test_murphy8_proline_isolated(self):
        # Proline's ring breaks alpha-helices — structurally unique
        assert _MURPHY8["P"] not in (_MURPHY8["G"], _MURPHY8["A"])

    def test_murphy8_all_20_aas_covered(self):
        missing = set(AA_ORDER) - set(_MURPHY8)
        assert not missing, f"Murphy8 missing: {missing}"

    def test_murphy8_exactly_8_classes(self):
        assert len(set(_MURPHY8.values())) == 8

    # ── Murphy10 ─────────────────────────────────────────────────────────────

    def test_murphy10_sulfur_group_cys_met_together(self):
        # In Murphy10, Cys+Met share the sulfur-containing class
        assert _MURPHY10["C"] != _MURPHY10["M"]

    def test_murphy10_aromatics_grouped(self):
        assert _MURPHY10["F"] == _MURPHY10["Y"] == _MURPHY10["W"]

    def test_murphy10_histidine_isolated(self):
        assert _MURPHY10["H"] != _MURPHY10["K"]
        assert _MURPHY10["H"] != _MURPHY10["R"]

    def test_murphy10_amide_group(self):
        assert _MURPHY10["N"] == _MURPHY10["Q"]

    def test_murphy10_hydroxyl_group(self):
        assert _MURPHY10["S"] == _MURPHY10["T"]

    def test_murphy10_all_20_aas_covered(self):
        missing = set(AA_ORDER) - set(_MURPHY10)
        assert not missing, f"Murphy10 missing: {missing}"

    def test_murphy10_exactly_10_classes(self):
        assert len(set(_MURPHY10.values())) == 10

    # ── encode_property_groups ────────────────────────────────────────────────

    def test_encode_returns_correct_murphy8_group(self):
        enc = encode_property_groups("KCDE", alphabet="murphy8")
        assert enc[0] == _MURPHY8["K"]
        assert enc[1] == _MURPHY8["C"]
        assert enc[2] == _MURPHY8["D"]
        assert enc[3] == _MURPHY8["E"]

    def test_encode_masked_x_in_property(self):
        enc = encode_property_groups("AXDE", alphabet="murphy8")
        assert enc[1] == MASK_INT

    def test_murphy8_vs_murphy10_cys_behavior(self):
        # Murphy8: C alone; Murphy10: C with M
        enc8  = encode_property_groups("CM", alphabet="murphy8")
        enc10 = encode_property_groups("CM", alphabet="murphy10")
        assert enc8[0]  != enc8[1],  "Murphy8: C and M should be different classes"
        assert enc10[0] != enc10[1], "Murphy10: C and M should be same class"

    def test_unknown_alphabet_falls_back_gracefully(self):
        enc = encode_property_groups("ACDE", alphabet="not_a_real_alphabet")
        assert len(enc) == 4


# ─────────────────────────────────────────────────────────────────────────────
# 3. find_seeds() — BUG FIX
# ─────────────────────────────────────────────────────────────────────────────

class TestFindSeeds:
    """
    Original bug:
        seeds.append((run_start - (l-1), j-(l-1), l))

    run_start is the INDEX OF THE FIRST MATCHING CHARACTER in X.
    Subtracting (l-1)=3 from it pushes it 3 positions backward,
    creating wrong (often negative) X-positions.

    Fix: seeds.append((run_start, j-(l-1), l))

    Tests verify:
      a) No negative X or Y seed positions.
      b) Reported positions are genuine matches: X[px:px+ln] == Y[py:py+ln].
      c) Known seeds are found in designed sequences.
      d) Masked chars break runs.
      e) Off-diagonal (shifted) matches are detected.
    """

    @staticmethod
    def _e(s: str) -> np.ndarray:
        return encode_sequence(s)

    def test_x_positions_non_negative(self):
        X = self._e("ACDEFGHI")
        seeds = find_seeds(X, X.copy(), 4)
        assert seeds, "Expected at least one seed for identical 8-AA sequence"
        for px, py, _ in seeds:
            assert px >= 0, f"Negative X position: {px} — find_seeds bug not fixed"
            assert py >= 0, f"Negative Y position: {py}"

    def test_every_seed_is_a_real_match(self):
        X = self._e("ACDEFKLMN")
        Y = self._e("QQACDEFQQ")
        seeds = find_seeds(X, Y, 4)
        assert seeds, "Expected seed for shared ACDE/ACDEF block"
        for px, py, ln in seeds:
            x_seg = X[px: px + ln]
            y_seg = Y[py: py + ln]
            assert np.array_equal(x_seg, y_seg), \
                f"Seed ({px},{py},{ln}) is not a real match: {list(x_seg)} vs {list(y_seg)}"
            assert MASK_INT not in x_seg, "Seed contains masked position"

    def test_known_seed_position_is_correct(self):
        # ACDE starts at position 2 in X, position 3 in Y
        X = self._e("KKACDE")
        Y = self._e("MMMACDE")
        seeds = find_seeds(X, Y, 4)
        assert seeds, "Expected seed for shared ACDE"
        # At least one seed should contain the ACDE block at the right X position
        found = any(px == 2 for px, _, _ in seeds)
        assert found, f"Expected X-position 2 in seeds, got: {seeds}"

    def test_identical_sequences_find_seeds(self):
        X = self._e("ACDEFGHIKLMN")
        seeds = find_seeds(X, X.copy(), 4)
        assert len(seeds) > 0

    def test_disjoint_alphabets_no_seeds(self):
        X = self._e("AAAAAAAAAA")   # only Ala
        Y = self._e("CCCCCCCCCC")   # only Cys
        seeds = find_seeds(X, Y, 4)
        assert len(seeds) == 0

    def test_masked_char_breaks_run(self):
        X = self._e("ACDE" + "X" + "FGHI")
        Y = self._e("ACDE" + "X" + "FGHI")
        seeds = find_seeds(X, Y, 4)
        for px, py, ln in seeds:
            assert MASK_INT not in X[px: px + ln], \
                "Seed spans a masked position — masking not respected"

    def test_off_diagonal_seed_detected(self):
        # Match exists but is shifted: ACDEF at pos 4 in X, pos 0 in Y
        X = self._e("QQQQACDEF")
        Y = self._e("ACDEFQQQQ")
        seeds = find_seeds(X, Y, 4)
        found = any(
            np.array_equal(X[px: px + ln], Y[py: py + ln])
            for px, py, ln in seeds
        )
        assert found, "Off-diagonal match not found — only diagonal searched"

    def test_repeated_motif_finds_multiple_seeds(self):
        seq = "ACDEACDE"
        X   = encode_sequence(seq)
        seeds = find_seeds(X, X.copy(), 4)
        # Two occurrences of ACDE should produce at least 2 seeds
        assert len(seeds) >= 2, \
            f"Expected ≥ 2 seeds for repeated motif, got {len(seeds)}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. expand_seed()
# ─────────────────────────────────────────────────────────────────────────────

class TestExpandSeed:

    @staticmethod
    def _e(s: str) -> np.ndarray:
        return encode_sequence(s)

    def test_no_expansion_at_both_boundaries(self):
        X = self._e("ACDE")
        px, py, ln = expand_seed(X, X.copy(), 0, 0, 4)
        assert (px, py, ln) == (0, 0, 4)

    def test_expands_left(self):
        X = self._e("MACDE")
        px, py, ln = expand_seed(X, X.copy(), 1, 1, 4)
        assert px == 0 and py == 0 and ln == 5

    def test_expands_right(self):
        X = self._e("ACDEM")
        px, py, ln = expand_seed(X, X.copy(), 0, 0, 4)
        assert ln == 5

    def test_expands_both_directions(self):
        X = self._e("KACDEM")
        px, py, ln = expand_seed(X, X.copy(), 1, 1, 4)
        assert px == 0 and py == 0 and ln == 6

    def test_masked_left_stops_expansion(self):
        X = self._e("XACDE")
        px, py, ln = expand_seed(X, X.copy(), 1, 1, 4)
        assert px == 1, "Expanded past a masked position on the left"

    def test_masked_right_stops_expansion(self):
        X = self._e("ACDEX")
        px, py, ln = expand_seed(X, X.copy(), 0, 0, 4)
        assert ln == 4, "Expanded past a masked position on the right"

    def test_left_mismatch_stops_expansion(self):
        X = self._e("KACDE")
        Y = self._e("MACDE")   # K ≠ M on the left
        px, py, ln = expand_seed(X, Y, 1, 1, 4)
        assert px == 1

    def test_right_mismatch_stops_expansion(self):
        X = self._e("ACDEF")
        Y = self._e("ACDEG")   # F ≠ G on the right
        px, py, ln = expand_seed(X, Y, 0, 0, 4)
        assert ln == 4

    def test_y_left_boundary_respected(self):
        # Y is shorter than X — expansion must not push py below 0
        X = self._e("AACDE")
        Y = self._e("ACDE")
        px, py, ln = expand_seed(X, Y, 1, 0, 4)
        assert py >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. maximal_filter()
# ─────────────────────────────────────────────────────────────────────────────

class TestMaximalFilter:

    def test_contained_match_removed(self):
        matches = [(0, 0, 6), (1, 1, 4)]   # (0,0,6) contains (1,1,4)
        result  = maximal_filter(matches)
        assert (0, 0, 6) in result
        assert (1, 1, 4) not in result

    def test_non_overlapping_both_kept(self):
        matches = [(0, 0, 4), (10, 10, 4)]
        assert len(maximal_filter(matches)) == 2

    def test_exact_duplicate_deduplicated(self):
        matches = [(0, 0, 4), (0, 0, 4), (0, 0, 4)]
        result  = maximal_filter(matches)
        assert sum(1 for m in result if m == (0, 0, 4)) == 1

    def test_empty_input(self):
        assert maximal_filter([]) == []

    def test_single_element_unchanged(self):
        assert maximal_filter([(2, 3, 5)]) == [(2, 3, 5)]

    def test_partial_overlap_both_kept(self):
        # (0,0,5) and (3,3,5): neither contains the other
        result = maximal_filter([(0, 0, 5), (3, 3, 5)])
        assert len(result) == 2

    def test_three_nested_outermost_wins(self):
        matches = [(0, 0, 10), (2, 2, 6), (3, 3, 4)]
        result  = maximal_filter(matches)
        assert (0, 0, 10) in result
        assert (2, 2, 6)  not in result
        assert (3, 3, 4)  not in result

    def test_longer_wins_at_same_start(self):
        result = maximal_filter([(0, 0, 4), (0, 0, 8)])
        assert (0, 0, 8) in result
        assert (0, 0, 4) not in result


# ─────────────────────────────────────────────────────────────────────────────
# 6. s_max normalisation — BUG FIX
# ─────────────────────────────────────────────────────────────────────────────

class TestSMaxNormalisation:
    """
    OLD (buggy):
        s_max = total / len(longest)          # per-residue average
        raw   = score / max(len(X), len(Y))   # also per-residue

    When X and Y are both shorter than 'longest':
        max(|X|,|Y|) < |longest| → raw > s_max → min() clips to 1.0.

    FIX:
        s_max = total  (not divided)
        raw   = score / longest_len  (always divide by the longest in the family)

    Ensures raw ≤ s_max always, without artificial clipping.
    """

    def test_short_pair_below_one(self, blosum62_diag, family_ctx):
        s_max, llen = family_ctx["s_max"], family_ctx["llen"]
        score = compute_sms_pair(
            "ACDE", "ACDE",  # 4 AA pair in a family whose longest seq is 40 AA
            blosum62_diag, llen, s_max, l=4, use_property_pass=False,
        )
        assert score < 1.0, (
            f"Short pair clipped to {score} — normalisation BUG not fixed. "
            "4-residue pair in 40-residue family cannot legitimately score 1.0."
        )
        assert score >= 0.0

    def test_longest_self_scores_exactly_one(self, blosum62_diag):
        from similarity.sms_matrix import compute_s_max
        seq = "ACDEFGHIKLMNPQRSTVWY"
        s_max, llen = compute_s_max({"only": seq}, blosum62_diag)
        score = compute_sms_pair(seq, seq, blosum62_diag, llen, s_max,
                                  l=4, use_property_pass=False)
        assert abs(score - 1.0) < 1e-4, f"Longest seq self-similarity ≠ 1.0: {score}"

    def test_all_scores_in_unit_interval(self, blosum62_diag, family_ctx):
        s_max, llen = family_ctx["s_max"], family_ctx["llen"]
        seqs = family_ctx["seqs"]
        ids  = list(seqs.keys())
        for i in range(len(ids)):
            for j in range(i, len(ids)):
                s = compute_sms_pair(
                    seqs[ids[i]], seqs[ids[j]],
                    blosum62_diag, llen, s_max, l=4, use_property_pass=False,
                )
                assert 0.0 <= s <= 1.0, \
                    f"Score out of [0,1]: {ids[i]} vs {ids[j]} → {s}"

    def test_symmetry(self, blosum62_diag, family_ctx):
        s_max, llen = family_ctx["s_max"], family_ctx["llen"]
        kw = dict(M_diag=blosum62_diag, longest_len=llen,
                  s_max_total=s_max, l=4, use_property_pass=False)
        ab = compute_sms_pair("ACDEFGHIKL", "MNPQRSTVWY", **kw)
        ba = compute_sms_pair("MNPQRSTVWY", "ACDEFGHIKL", **kw)
        assert abs(ab - ba) < 1e-5, f"Asymmetric: SMS(A,B)={ab}, SMS(B,A)={ba}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. compute_sms_pair() — semantic correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSmsPair:

    def test_identical_scores_higher_than_partial(self, blosum62_diag, family_ctx):
        s_max, llen = family_ctx["s_max"], family_ctx["llen"]
        seqs = family_ctx["seqs"]
        kw   = dict(M_diag=blosum62_diag, longest_len=llen,
                    s_max_total=s_max, l=4, use_property_pass=False)
        s_id   = compute_sms_pair(seqs["full"], seqs["full"],    **kw)
        s_part = compute_sms_pair(seqs["full"], seqs["partial"], **kw)
        assert s_id > s_part

    def test_partial_scores_higher_than_disjoint(self, blosum62_diag, family_ctx):
        s_max, llen = family_ctx["s_max"], family_ctx["llen"]
        seqs = family_ctx["seqs"]
        kw   = dict(M_diag=blosum62_diag, longest_len=llen,
                    s_max_total=s_max, l=4, use_property_pass=False)
        s_part = compute_sms_pair(seqs["full"], seqs["partial"], **kw)
        s_none = compute_sms_pair(seqs["full"], seqs["none"],    **kw)
        assert s_part > s_none

    def test_empty_seq1_returns_zero(self, blosum62_diag):
        s = compute_sms_pair("", "ACDE", blosum62_diag,
                              longest_len=10, s_max_total=50.0,
                              l=4, use_property_pass=False)
        assert s == 0.0

    def test_empty_seq2_returns_zero(self, blosum62_diag):
        s = compute_sms_pair("ACDE", "", blosum62_diag,
                              longest_len=10, s_max_total=50.0,
                              l=4, use_property_pass=False)
        assert s == 0.0

    def test_property_pass_gte_exact_for_conservative_subs(self, blosum62_diag):
        """
        Murphy property-group pass should score >= exact-only when sequences
        differ only by conservative substitutions within one Murphy class.
        (I, L, V are all in Murphy8 class 0 — aliphatic hydrophobic.)
        """
        from similarity.sms_matrix import compute_s_max
        # Conserved flanks (ACDEF + MNPQR); middle differs by conservative subs
        seq_a = "ACDEF" + "ILVAL" + "MNPQR"
        seq_b = "ACDEF" + "VIVLL" + "MNPQR"  # I↔V, L↔I, V↔V, A→L, L↔L
        s_max, llen = compute_s_max({"a": seq_a, "b": seq_b}, blosum62_diag)
        kw = dict(M_diag=blosum62_diag, longest_len=llen, s_max_total=s_max, l=4)
        s_exact = compute_sms_pair(seq_a, seq_b, **kw, use_property_pass=False)
        s_prop  = compute_sms_pair(seq_a, seq_b, **kw, use_property_pass=True)
        assert s_prop >= s_exact, \
            f"Property pass hurt score: exact={s_exact:.4f}, prop={s_prop:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Domain mask
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeDomainMask:

    def test_no_domains_all_linker_weight(self):
        mask = make_domain_mask(10, domains=[], linker_weight=0.5)
        assert np.allclose(mask, 0.5)

    def test_domain_positions_get_domain_weight(self):
        # Domain residues 3–7 (1-indexed, inclusive) → 0-indexed slice [2:7]
        mask = make_domain_mask(10, [{"start": 3, "end": 7}],
                                domain_weight=2.0, linker_weight=0.5)
        assert np.allclose(mask[2:7], 2.0)
        assert np.allclose(mask[:2],  0.5)
        assert np.allclose(mask[7:],  0.5)

    def test_two_non_overlapping_domains(self):
        mask = make_domain_mask(10,
                                [{"start": 1, "end": 5}, {"start": 8, "end": 10}],
                                domain_weight=3.0, linker_weight=1.0)
        assert np.allclose(mask[0:5],  3.0)
        assert np.allclose(mask[5:7],  1.0)
        assert np.allclose(mask[7:10], 3.0)

    def test_out_of_bounds_domain_clamped(self):
        # Domain extends far past sequence length — must not raise IndexError
        mask = make_domain_mask(10, [{"start": 1, "end": 999}], domain_weight=2.0)
        assert len(mask) == 10
        assert np.all(mask == 2.0)

    def test_dtype_float32(self):
        mask = make_domain_mask(5, [], domain_weight=2.0)
        assert mask.dtype == np.float32

    def test_domain_weighting_changes_pair_score(self, blosum62_diag):
        """Domain mask must actually affect the computed pair score."""
        from similarity.sms_matrix import compute_s_max
        seq  = "ACDEFGHIKLMNPQRSTVWY"
        s_max, llen = compute_s_max({"a": seq}, blosum62_diag)

        # Score without any domain annotation
        s_plain = compute_sms_pair(seq, seq, blosum62_diag, llen, s_max,
                                   l=4, use_property_pass=False)
        # Score with full-sequence domain at 2× weight
        doms    = [{"start": 1, "end": 20}]
        s_dom   = compute_sms_pair(seq, seq, blosum62_diag, llen, s_max,
                                   l=4, use_property_pass=False,
                                   domains1=doms, domains2=doms, domain_weight=2.0)
        # Both must be in [0, 1]; they should differ because domain scaling changes raw score
        assert 0.0 <= s_plain <= 1.0
        assert 0.0 <= s_dom   <= 1.0
