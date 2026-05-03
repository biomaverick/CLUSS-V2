"""
tests/test_pipeline_small.py
══════════════════════════════
End-to-end integration test of the full CLUSS+ pipeline on a synthetic
12-sequence dataset with a known ground-truth clustering.

Sequence design
───────────────
Three charge-class groups with strictly disjoint 6-AA alphabets:

  Group A — aliphatic/hydrophobic  : {I, L, V, A, G, M}
  Group B — cationic/polar         : {K, R, H, N, Q, T}
  Group C — anionic/aromatic       : {D, E, P, Y, W, C}

Each group shares an 8-residue conserved core (unique to that group's alphabet)
embedded in varied 12-residue flanks (also from the group alphabet). This gives:
  - Within-group SMS similarity ≈ 0.55–0.80 (core matches, flanks vary)
  - Between-group SMS similarity = 0.00 (disjoint alphabets, no k-mer in common)

Disjoint alphabets ensure zero inter-group similarity even with the property-group
pass disabled. The property pass is tested separately in test_sms_engine.py.

The 0.0 inter-group cosimilarity values are crisply separated from within-group
values by the Otsu threshold, giving clean 3-cluster extraction.

Run with
────────
  cd cluss_plus/
  python -m pytest tests/test_pipeline_small.py -v
"""

import sys
import os
import pytest
import numpy as np


from preprocessing.fasta_parser    import parse_fasta, validate_sequences
from preprocessing.complexity_mask import mask_all_sequences, mask_low_complexity
from similarity.sms_matrix         import build_sms_matrix, compute_s_max
from tree.phylo_tree               import build_phylo_tree, assign_depths
from clustering.cosimilarity       import (compute_leaf_weights,
                                           compute_node_weights,
                                           compute_cosimilarity)
from clustering.boundary_detector  import detect_boundaries
from clustering.cluster_extractor  import extract_clusters, collect_leaves
from evaluation.q_measure          import compute_q_measure


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────
#
# Alphabet per group (strictly disjoint — no shared amino acid across groups):
#   A: {I, L, V, A, G, M}   — 6 AAs, all non-polar/aliphatic
#   B: {K, R, H, N, Q, T}   — 6 AAs, all charged/polar
#   C: {D, E, P, Y, W, C}   — 6 AAs, all anionic/aromatic/special
#
# Sequence structure: 6 aa left-flank + 8 aa core + 6 aa right-flank = 20 aa
#   Core is UNIQUE per group and SHARED across all members of that group.
#   Flanks vary between sequences (picked from same group alphabet).
#
# Property-pass note: Murphy8 class 1 = {F,Y,W}. Group C contains W and Y
# but NOT F. Group A contains NO aromatic AAs. Group B contains NO aromatics.
# → even with property_pass=True there is no inter-group match.
# However, we use use_property_pass=False in the fixture to keep the test
# focused on exact-match SMS behaviour (property pass has its own unit tests).

CORE_A = "ILVAGMIL"    # 8-residue core unique to group A
CORE_B = "KRHNTQKR"    # 8-residue core unique to group B
CORE_C = "DEPYWCDE"    # 8-residue core unique to group C

SEQUENCES: dict[str, str] = {
    # Group A — aliphatic/hydrophobic (alphabet: I L V A G M)
    "A1": "ILMVGA" + CORE_A + "VMGLAI",
    "A2": "GALIVM" + CORE_A + "MILAGV",
    "A3": "VMGAIL" + CORE_A + "GALVIM",
    "A4": "AMVLGI" + CORE_A + "IVGAML",
    # Group B — cationic/polar (alphabet: K R H N Q T)
    "B1": "KTNHRQ" + CORE_B + "QRHNKT",
    "B2": "RQKNTK" + CORE_B + "NKQTHR",
    "B3": "HNQTRK" + CORE_B + "TRHQNK",
    "B4": "QKRHTN" + CORE_B + "HNTRKQ",
    # Group C — anionic/aromatic (alphabet: D E P Y W C)
    "C1": "DEYWPC" + CORE_C + "PCYWDE",
    "C2": "YWCPED" + CORE_C + "EDCPYW",
    "C3": "PCDYWE" + CORE_C + "WYPECD",
    "C4": "WEPYCD" + CORE_C + "CDEPWY",
}

# Sanity: all sequences should be exactly 20 residues
assert all(len(s) == 20 for s in SEQUENCES.values()), "Sequence length mismatch"

# Sanity: alphabets are disjoint
_alpha_A = set("ILVAGM")
_alpha_B = set("KRHNTQ")
_alpha_C = set("DEPYWC")
assert _alpha_A.isdisjoint(_alpha_B)
assert _alpha_A.isdisjoint(_alpha_C)
assert _alpha_B.isdisjoint(_alpha_C)

REFERENCE: dict[str, str] = {
    **{f"A{i}": "hydrophobic" for i in range(1, 5)},
    **{f"B{i}": "cationic"    for i in range(1, 5)},
    **{f"C{i}": "anionic"     for i in range(1, 5)},
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures (module-scoped so the slow SMS build runs only once)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_fasta(tmp_path_factory):
    """Write the toy dataset to a temporary FASTA file."""
    path = str(tmp_path_factory.mktemp("data") / "toy.fasta")
    with open(path, "w") as fh:
        for sid, seq in SEQUENCES.items():
            fh.write(f">{sid}\n{seq}\n")
    return path


@pytest.fixture(scope="module")
def validated_seqs():
    return validate_sequences(dict(SEQUENCES), min_len=10)


@pytest.fixture(scope="module")
def sms_result(validated_seqs):
    """Build the 12×12 SMS matrix (exact-match only for clean test isolation)."""
    ids, S = build_sms_matrix(
        validated_seqs,
        matrix_names      = ["BLOSUM62"],
        matrix_weights    = [1.0],
        l                 = 4,
        use_property_pass = False,   # property pass tested separately
        n_jobs            = 1,
    )
    return ids, S


@pytest.fixture(scope="module")
def pipeline(sms_result):
    """Run all clustering stages and return full artefact dict."""
    ids, S = sms_result
    root, nodes   = build_phylo_tree(S, ids, method="upgma")
    leaf_weights  = compute_leaf_weights(root)
    node_weights  = compute_node_weights(root, leaf_weights)
    cosim         = compute_cosimilarity(root, node_weights)
    cut_nodes     = detect_boundaries(cosim, method="otsu")
    clusters, orphans, cluster_ids = extract_clusters(root, cut_nodes, min_size=2)
    return {
        "ids": ids, "S": S, "root": root, "nodes": nodes,
        "leaf_weights": leaf_weights, "node_weights": node_weights,
        "cosim": cosim, "cut_nodes": cut_nodes,
        "clusters": clusters, "orphans": orphans, "cluster_ids": cluster_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: FASTA parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestFastaParsing:

    def test_parses_all_12_sequences(self, tmp_fasta):
        seqs = parse_fasta(tmp_fasta)
        assert len(seqs) == 12

    def test_all_sequence_ids_present(self, tmp_fasta):
        seqs = parse_fasta(tmp_fasta)
        for sid in SEQUENCES:
            assert sid in seqs, f"Missing: {sid}"

    def test_sequence_content_intact(self, tmp_fasta):
        seqs = parse_fasta(tmp_fasta)
        for sid, expected in SEQUENCES.items():
            assert seqs[sid] == expected.upper()

    def test_validation_passes_all_12(self, validated_seqs):
        assert len(validated_seqs) == 12

    def test_min_len_filter_drops_short_sequences(self):
        seqs     = {"short": "ACDE", "ok": "ACDEFGHIKLMNPQRSTVWY"}
        filtered = validate_sequences(seqs, min_len=10)
        assert "short" not in filtered
        assert "ok" in filtered

    def test_invalid_char_filter_drops_sequence(self):
        seqs     = {"bad": "ACDE*FGHI", "ok": "ACDEFGHIKL"}
        filtered = validate_sequences(seqs, min_len=5)
        assert "bad" not in filtered
        assert "ok" in filtered


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: LCR masking
# ─────────────────────────────────────────────────────────────────────────────

class TestLCRMasking:

    def test_masking_preserves_length(self, validated_seqs):
        masked = mask_all_sequences(validated_seqs)
        for sid, seq in validated_seqs.items():
            assert len(masked[sid]) == len(seq), \
                f"{sid}: length changed after masking"

    def test_homopolymer_gets_masked(self):
        masked = mask_low_complexity("A" * 30, window=12, k1=1.8, k2=2.5)
        assert "X" in masked, "30-residue poly-Ala should be masked by SEG"

    def test_high_entropy_sequence_not_masked(self):
        seq    = "ACDEFGHIKLMNPQRSTVWY" * 2   # all 20 AAs — max entropy
        masked = mask_low_complexity(seq, window=12, k1=1.8, k2=2.5)
        assert "X" not in masked, \
            "High-entropy sequence should not be masked"

    def test_toy_sequences_survive_masking_intact(self, validated_seqs):
        """
        Toy sequences use ≥ 6 unique AAs within any 12-residue window
        → entropy > 2.3 bits >> SEG trigger threshold of 1.8 bits.
        No residue should be masked.
        """
        masked = mask_all_sequences(validated_seqs)
        for sid, seq in masked.items():
            assert "X" not in seq, \
                f"{sid}: {seq.count('X')} residues unexpectedly masked"

    def test_two_threshold_triggers_then_extends(self):
        """
        SEG should trigger on a low-entropy core and extend outward.
        A 6-residue poly-Ala core flanked by diverse residues should mask
        the core but leave the diverse flanks unmasked.
        """
        diverse = "ACDEFG"       # high entropy
        core    = "AAAAAA"       # very low entropy
        seq     = diverse + core + diverse
        masked  = mask_low_complexity(seq, window=6, k1=1.0, k2=3.0)
        # The core poly-A should be masked
        assert "X" in masked[6:12], \
            f"Low-entropy core not masked: '{masked[6:12]}'"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: SMS similarity matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestSMSMatrix:

    def test_matrix_shape(self, sms_result):
        _, S = sms_result
        assert S.shape == (12, 12)

    def test_matrix_is_symmetric(self, sms_result):
        _, S = sms_result
        assert np.allclose(S, S.T, atol=1e-5), "Matrix is not symmetric"

    def test_diagonal_is_one(self, sms_result):
        _, S = sms_result
        assert np.allclose(np.diag(S), 1.0, atol=1e-4)

    def test_all_values_in_unit_interval(self, sms_result):
        _, S = sms_result
        assert S.min() >= -1e-6
        assert S.max() <= 1.0 + 1e-6

    def test_inter_group_similarity_is_zero(self, sms_result):
        """
        Disjoint alphabets guarantee zero shared k-mers.
        Every cross-group pair must have SMS score = 0.
        """
        ids, S = sms_result
        idx    = {sid: i for i, sid in enumerate(ids)}
        groups = {
            "A": ["A1", "A2", "A3", "A4"],
            "B": ["B1", "B2", "B3", "B4"],
            "C": ["C1", "C2", "C3", "C4"],
        }
        group_pairs = [("A", "B"), ("A", "C"), ("B", "C")]
        for ga, gb in group_pairs:
            for sa in groups[ga]:
                for sb in groups[gb]:
                    score = float(S[idx[sa], idx[sb]])
                    assert score == 0.0, \
                        f"Inter-group ({ga}/{gb}) score {sa}↔{sb} = {score:.4f} ≠ 0"

    def test_intra_group_similarity_above_inter(self, sms_result):
        """
        Mean within-group similarity must be strictly greater than
        mean between-group similarity (= 0.0 for disjoint alphabets).
        """
        ids, S = sms_result
        idx    = {sid: i for i, sid in enumerate(ids)}
        groups = [["A1","A2","A3","A4"], ["B1","B2","B3","B4"], ["C1","C2","C3","C4"]]

        for group in groups:
            within = [S[idx[a], idx[b]]
                      for i, a in enumerate(group)
                      for b in group[i + 1:]]
            mean_within = np.mean(within)
            assert mean_within > 0.0, \
                f"Group {group[0][0]} has zero within-group similarity — shared core missing"

    def test_core_motif_drives_similarity(self, sms_result):
        """
        A1 and A2 share the ILVAGMIL core — their pairwise score must be > 0.
        """
        ids, S = sms_result
        idx    = {sid: i for i, sid in enumerate(ids)}
        score  = float(S[idx["A1"], idx["A2"]])
        assert score > 0.0, \
            f"A1↔A2 should share core k-mers but score = {score}"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Phylogenetic tree
# ─────────────────────────────────────────────────────────────────────────────

class TestPhylogeneticTree:

    def test_all_12_leaves_present(self, pipeline):
        leaves = collect_leaves(pipeline["root"])
        assert set(leaves) == set(SEQUENCES)

    def test_root_has_12_leaves(self, pipeline):
        assert pipeline["root"].n_leaves == 12

    def test_root_has_no_parent(self, pipeline):
        assert pipeline["root"].parent is None

    def test_all_leaf_nodes_have_seq_id(self, pipeline):
        for node in pipeline["nodes"].values():
            if node.is_leaf:
                assert node.seq_id in SEQUENCES

    def test_branch_lengths_non_negative(self, pipeline):
        for node in pipeline["nodes"].values():
            assert node.branch_length >= 0.0, \
                f"Negative branch length at node {node.id}"

    def test_depths_assigned_bug_fix(self, pipeline):
        """BUG FIX: node.depth must be assigned after tree build (was missing)."""
        for node in pipeline["nodes"].values():
            assert hasattr(node, "depth"), f"Node {node.id} missing 'depth'"
            assert node.depth >= 0

    def test_root_depth_zero(self, pipeline):
        assert pipeline["root"].depth == 0

    def test_all_leaves_depth_greater_than_zero(self, pipeline):
        for node in pipeline["nodes"].values():
            if node.is_leaf:
                assert node.depth > 0, \
                    f"Leaf {node.seq_id} has depth 0"

    def test_nj_tree_produces_same_leaves(self, sms_result):
        ids, S  = sms_result
        root_nj, _ = build_phylo_tree(S, ids, method="nj")
        assert root_nj.n_leaves == 12
        assert set(collect_leaves(root_nj)) == set(ids)

    def test_nj_depths_assigned(self, sms_result):
        ids, S  = sms_result
        root_nj, nodes_nj = build_phylo_tree(S, ids, method="nj")
        for node in nodes_nj.values():
            assert node.depth >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Co-similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestCoSimilarity:

    def test_cosim_computed_for_all_internal_nodes(self, pipeline):
        internal_ids = {nid for nid, n in pipeline["nodes"].items()
                        if not n.is_leaf}
        assert set(pipeline["cosim"].keys()) == internal_ids

    def test_cosim_non_negative(self, pipeline):
        for nid, val in pipeline["cosim"].items():
            assert val >= 0.0, f"Negative co-similarity at node {nid}: {val}"

    def test_all_leaves_have_weight(self, pipeline):
        leaf_ids = {nid for nid, n in pipeline["nodes"].items() if n.is_leaf}
        assert set(pipeline["leaf_weights"].keys()) == leaf_ids

    def test_leaf_weights_non_negative(self, pipeline):
        for lid, w in pipeline["leaf_weights"].items():
            assert w >= 0.0, f"Negative leaf weight at {lid}: {w}"

    def test_inter_group_cosim_lower_than_intra(self, pipeline):
        """
        Nodes that merge two different groups (inter-group merges) must have
        lower co-similarity than nodes that merge within a group.
        """
        cosim = pipeline["cosim"]
        vals  = sorted(cosim.values())
        # The two zeros in cosim correspond to the two inter-group merges
        assert vals[0] == 0.0, "Expected at least one zero co-similarity (inter-group)"
        # At least some non-zero values should exist (within-group merges)
        assert vals[-1] > 0.0, "All co-similarity values are zero — something wrong"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Boundary detection
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryDetection:

    @pytest.mark.parametrize("method", ["otsu", "gmm", "kneedle"])
    def test_boundary_returns_a_set_of_node_ids(self, pipeline, method):
        cosim     = pipeline["cosim"]
        cut_nodes = detect_boundaries(cosim, method=method)
        assert isinstance(cut_nodes, set)
        assert cut_nodes.issubset(set(cosim.keys()))

    def test_otsu_marks_at_least_one_cut(self, pipeline):
        cut_nodes = detect_boundaries(pipeline["cosim"], method="otsu")
        assert len(cut_nodes) >= 1

    def test_otsu_marks_the_zero_cosim_nodes(self, pipeline):
        """
        The two inter-group merges have co-similarity = 0.0 and must always
        be classified as cut nodes by any reasonable boundary method.
        """
        cosim     = pipeline["cosim"]
        cut_nodes = detect_boundaries(cosim, method="otsu")
        zero_nodes = {nid for nid, val in cosim.items() if val == 0.0}
        for nid in zero_nodes:
            assert nid in cut_nodes, \
                f"Node {nid} has cosim=0 but was NOT marked as a cut node"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: Cluster extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterExtraction:

    def test_all_sequences_accounted_for(self, pipeline):
        """Every sequence must be in exactly one cluster or in orphans."""
        all_assigned = (
            [sid for c in pipeline["clusters"] for sid in c]
            + pipeline["orphans"]
        )
        assert set(all_assigned) == set(SEQUENCES)

    def test_no_sequence_in_two_clusters(self, pipeline):
        all_sids = [sid for c in pipeline["clusters"] for sid in c]
        assert len(all_sids) == len(set(all_sids)), \
            "A sequence appears in more than one cluster"

    def test_produces_at_least_two_clusters(self, pipeline):
        assert len(pipeline["clusters"]) >= 2, \
            f"Expected ≥ 2 clusters for 3 disjoint groups, got {len(pipeline['clusters'])}"

    def test_clusters_meet_min_size(self, pipeline):
        for i, c in enumerate(pipeline["clusters"]):
            assert len(c) >= 2, f"Cluster {i} below min_size=2"

    def test_no_cluster_mixes_all_three_groups(self, pipeline):
        for i, cluster in enumerate(pipeline["clusters"]):
            groups = {REFERENCE[sid] for sid in cluster if sid in REFERENCE}
            assert len(groups) < 3, \
                f"Cluster {i} mixes all three charge classes: {groups}"

    def test_same_group_sequences_land_in_same_cluster(self, pipeline):
        """
        Since inter-group similarity is exactly 0, all four sequences of each
        group must appear in the same cluster (no intra-group splitting expected).
        """
        cid = pipeline["cluster_ids"]
        for group_letter in "ABC":
            member_clusters = {cid.get(f"{group_letter}{i}") for i in range(1, 5)}
            # Remove 'orphan' if any (should not happen with disjoint groups)
            cluster_ids = member_clusters - {"orphan"}
            assert len(cluster_ids) == 1, (
                f"Group {group_letter} members split across clusters: "
                f"{member_clusters}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: Q-measure
# ─────────────────────────────────────────────────────────────────────────────

class TestQMeasure:

    def test_q_in_valid_range(self, pipeline):
        Q = compute_q_measure(pipeline["clusters"], pipeline["orphans"], REFERENCE)
        assert 0.0 <= Q <= 100.0

    def test_q_above_75_for_clean_separation(self, pipeline):
        """
        With three disjoint-alphabet groups cleanly separated,
        Q-measure must be at least 75%.
        """
        Q = compute_q_measure(pipeline["clusters"], pipeline["orphans"], REFERENCE)
        assert Q >= 75.0, \
            f"Q = {Q:.1f}% — clustering of disjoint groups should score > 75%"

    def test_q_is_100_when_groups_perfectly_clustered(self):
        """
        Perfect clustering (each cluster = one reference group, zero orphans)
        must yield Q = 100.
        """
        perfect_clusters = [
            ["A1","A2","A3","A4"],
            ["B1","B2","B3","B4"],
            ["C1","C2","C3","C4"],
        ]
        Q = compute_q_measure(perfect_clusters, [], REFERENCE)
        assert abs(Q - 100.0) < 1e-4, f"Perfect clustering Q = {Q}"

    def test_q_clamped_to_zero_for_all_orphans(self):
        """BUG FIX: Q must not go negative when all sequences are orphans."""
        Q = compute_q_measure([], list(SEQUENCES.keys()), REFERENCE)
        assert Q == 0.0, f"All-orphan Q = {Q} (should be 0)"

    def test_q_with_uniform_weights_equals_unweighted(self, pipeline):
        """With all taxon weights = 1.0, weighted Q must equal unweighted Q."""
        weights = {sid: 1.0 for sid in SEQUENCES}
        Q_unw = compute_q_measure(pipeline["clusters"], pipeline["orphans"],
                                  REFERENCE, taxon_weights=None)
        Q_w   = compute_q_measure(pipeline["clusters"], pipeline["orphans"],
                                  REFERENCE, taxon_weights=weights)
        assert abs(Q_unw - Q_w) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Stage 9: Output writer
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputWriter:

    @pytest.fixture
    def out_dir(self, tmp_path):
        return str(tmp_path / "output")

    def test_clusters_tsv_has_correct_row_count(self, pipeline, out_dir):
        from output.writer import write_cluster_tsv
        write_cluster_tsv(pipeline["clusters"], pipeline["orphans"],
                          annotations=None, out_dir=out_dir)
        path  = os.path.join(out_dir, "clusters.tsv")
        assert os.path.exists(path)
        lines = open(path).readlines()
        n_clustered = sum(len(c) for c in pipeline["clusters"])
        assert len(lines) == n_clustered + 1, \
            f"Expected {n_clustered+1} lines (header + data), got {len(lines)}"

    def test_orphans_tsv_created(self, pipeline, out_dir):
        from output.writer import write_cluster_tsv
        write_cluster_tsv(pipeline["clusters"], pipeline["orphans"],
                          annotations=None, out_dir=out_dir)
        assert os.path.exists(os.path.join(out_dir, "orphans.tsv"))

    def test_newick_file_valid(self, pipeline, out_dir):
        from output.writer import write_newick
        write_newick(pipeline["root"], out_dir)
        path = os.path.join(out_dir, "tree.nwk")
        assert os.path.exists(path)
        content = open(path).read().strip()
        assert content.endswith(";")
        assert "(" in content

    def test_fasta_output_all_sequences(self, pipeline, out_dir):
        from output.writer import write_cluster_fasta
        write_cluster_fasta(SEQUENCES, pipeline["clusters"],
                            pipeline["orphans"], out_dir, split=False)
        path    = os.path.join(out_dir, "clusters.fasta")
        assert os.path.exists(path)
        headers = [l for l in open(path) if l.startswith(">")]
        assert len(headers) == 12

    def test_fasta_headers_contain_cluster_tag(self, pipeline, out_dir):
        from output.writer import write_cluster_fasta
        write_cluster_fasta(SEQUENCES, pipeline["clusters"],
                            pipeline["orphans"], out_dir, split=False)
        path = os.path.join(out_dir, "clusters.fasta")
        for line in open(path):
            if line.startswith(">"):
                assert "cluster=" in line, \
                    f"Header missing 'cluster=' tag: {line.strip()}"

    def test_split_fasta_creates_per_cluster_directory(self, pipeline, out_dir):
        from output.writer import write_cluster_fasta
        write_cluster_fasta(SEQUENCES, pipeline["clusters"],
                            pipeline["orphans"], out_dir, split=True)
        sub = os.path.join(out_dir, "per_cluster")
        assert os.path.isdir(sub)
        fasta_files = [f for f in os.listdir(sub) if f.endswith(".fasta")]
        assert len(fasta_files) >= len(pipeline["clusters"])

    def test_checkpoint_numpy_roundtrip(self, out_dir):
        from output.writer import save_checkpoint, load_checkpoint
        S = np.eye(5, dtype=np.float32)
        save_checkpoint("mat", S, out_dir)
        loaded = load_checkpoint("mat", out_dir)
        assert loaded is not None
        assert np.allclose(S, loaded)

    def test_checkpoint_json_roundtrip(self, out_dir):
        from output.writer import save_checkpoint, load_checkpoint
        data = {"k": [1, 2, 3], "v": "hello"}
        save_checkpoint("meta", data, out_dir)
        assert load_checkpoint("meta", out_dir) == data

    def test_checkpoint_missing_returns_none(self, out_dir):
        from output.writer import load_checkpoint
        assert load_checkpoint("does_not_exist", out_dir) is None

    def test_html_report_written_and_valid(self, pipeline, out_dir):
        from output.writer import write_html_report
        write_html_report(
            clusters    = pipeline["clusters"],
            orphans     = pipeline["orphans"],
            metrics     = {"Q_measure": 90.0, "runtime_seconds": 1.5},
            annotations = None,
            enriched_go = None,
            run_meta    = {"mode": "sms", "fasta": "toy.fasta"},
            out_dir     = out_dir,
        )
        path = os.path.join(out_dir, "report.html")
        assert os.path.exists(path)
        content = open(path).read()
        assert "CLUSS+" in content
        assert "Cluster Table" in content
        assert str(len(pipeline["clusters"])) in content

    def test_summary_json_contains_metrics(self, pipeline, out_dir):
        from output.writer import write_summary_json
        import json
        metrics  = {"Q_measure": 90.0, "ARI": 0.85, "runtime_seconds": 2.1}
        run_meta = {"mode": "sms"}
        write_summary_json(run_meta, pipeline["clusters"],
                           pipeline["orphans"], metrics, out_dir)
        path = os.path.join(out_dir, "summary.json")
        assert os.path.exists(path)
        data = json.load(open(path))
        assert data["n_clusters"] == len(pipeline["clusters"])
        assert "Q_measure" in data["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
# Mocked network tests (CI-safe: no real HTTP requests)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import responses as resp_lib
    HAS_RESPONSES = True
except ImportError:
    HAS_RESPONSES = False

import pytest

MOCK_UNIPROT_RESPONSE = {
    "entryType": "UniProtKB reviewed (Swiss-Prot)",
    "primaryAccession": "P12345",
    "proteinDescription": {
        "recommendedName": {"fullName": {"value": "Test protein"}}
    },
    "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
    "genes": [{"geneName": {"value": "TPROT"}}],
    "uniProtKBCrossReferences": [{"database": "GO", "id": "GO:0005488"}],
}


@pytest.mark.skipif(not HAS_RESPONSES, reason="responses library not installed")
class TestAnnotationMocked:
    """Network-free tests for uniprot_fetcher using mocked HTTP responses."""

    @resp_lib.activate
    def test_fetch_uniprot_annotation_returns_dict(self):
        from annotation.uniprot_fetcher import fetch_uniprot_annotation
        resp_lib.add(
            resp_lib.GET,
            "https://rest.uniprot.org/uniprotkb/P12345.json",
            json=MOCK_UNIPROT_RESPONSE,
            status=200,
        )
        result = fetch_uniprot_annotation("P12345")
        assert isinstance(result, dict)
        assert result["accession"] == "P12345"
        assert result["organism"] == "Homo sapiens"
        assert result["reviewed"] is True
        assert "GO:0005488" in result["go_terms"]

    @resp_lib.activate
    def test_fetch_uniprot_404_returns_empty_stub(self):
        from annotation.uniprot_fetcher import fetch_uniprot_annotation
        resp_lib.add(
            resp_lib.GET,
            "https://rest.uniprot.org/uniprotkb/ZZZZZZ.json",
            status=404,
        )
        result = fetch_uniprot_annotation("ZZZZZZ")
        assert result["organism"] == "Unknown"
        assert result["accession"] is None

    @resp_lib.activate
    def test_fetch_all_annotations_no_accession(self):
        """Sequence IDs that are not UniProt accessions get empty stubs."""
        from annotation.uniprot_fetcher import fetch_all_annotations
        # GenBank-style IDs — no UniProt lookup expected
        results = fetch_all_annotations(
            ["AAA24053", "GenBank:BAD89079"],
            fetch_interpro=False,
        )
        assert "AAA24053" in results
        assert results["AAA24053"]["organism"] == "Unknown"


MOCK_IUPRED_RESPONSE = {
    "iupred_scores": [0.8, 0.3, 0.9, 0.1, 0.7]
}


@pytest.mark.skipif(not HAS_RESPONSES, reason="responses library not installed")
class TestIdrMaskingMocked:
    """Network-free test for IUPred3 IDR masking."""

    @resp_lib.activate
    def test_mask_disordered_regions_applies_threshold(self):
        from preprocessing.complexity_mask import mask_disordered_regions
        seq = "ACDEF"
        resp_lib.add(
            resp_lib.GET,
            "https://iupred3.elte.hu/iupred3API",
            json=MOCK_IUPRED_RESPONSE,
            status=200,
        )
        masked = mask_disordered_regions(seq, threshold=0.5)
        # Positions 0, 2, 4 (scores 0.8, 0.9, 0.7) should be masked
        assert masked[0] == "X"
        assert masked[1] == "C"   # score 0.3, not masked
        assert masked[2] == "X"
        assert masked[3] == "E"   # score 0.1, not masked
        assert masked[4] == "X"


# ─────────────────────────────────────────────────────────────────────────────
# Big-data smoke test (marked slow — only runs with: pytest -m slow)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_matrix_build_1000_sequences(tmp_path):
    """Smoke test: build_sms_matrix completes for N=1000 without OOM."""
    import random
    AA = "ACDEFGHIKLMNPQRSTVWY"
    random.seed(99)
    seqs = {f"seq_{i}": "".join(random.choices(AA, k=50)) for i in range(1000)}
    from similarity.sms_matrix import build_sms_matrix
    ids, S = build_sms_matrix(
        seqs,
        n_jobs=2,
        chunk_size=500,
        out_dir=str(tmp_path),
    )
    assert S.shape == (1000, 1000)
    assert np.allclose(np.diag(S), 1.0, atol=1e-4)
    assert S.min() >= 0.0
    assert S.max() <= 1.0 + 1e-6
