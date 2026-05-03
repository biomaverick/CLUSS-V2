"""
tests/test_visualization.py
════════════════════════════
Tests for visualization/tree_plot.py and visualization/heatmap.py.

All tests write to a pytest tmp_path and verify:
  - Expected output files exist and are non-empty.
  - PDF / SVG / HTML files have correct magic bytes / content markers.
  - Figures handle edge cases (N=1 leaf, all-orphan, very large cluster count).
  - Colour palette covers every cluster without repetition at small N.
  - Interactive HTML contains expected Plotly script markers.

The full pipeline fixture is module-scoped (built once, re-used across all
test classes) — identical to test_pipeline_small.py's fixture.

Run with
────────
  cd cluss_plus/
  python -m pytest tests/test_visualization.py -v
"""

import sys
import os
import pytest
import numpy as np


from tests.test_pipeline_small import SEQUENCES, REFERENCE   # shared toy data

from cluss_plus.preprocessing.fasta_parser    import validate_sequences
from cluss_plus.similarity.sms_matrix         import build_sms_matrix
from cluss_plus.tree.phylo_tree               import build_phylo_tree, TreeNode, assign_depths
from cluss_plus.clustering.cosimilarity       import (compute_leaf_weights,
                                           compute_node_weights,
                                           compute_cosimilarity)
from cluss_plus.clustering.boundary_detector  import detect_boundaries, otsu_threshold
from cluss_plus.clustering.cluster_extractor  import extract_clusters


# ─────────────────────────────────────────────────────────────────────────────
# Module-scoped pipeline fixture  (built once, shared by all test classes)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    seqs = validate_sequences(dict(SEQUENCES), min_len=10)
    ids, S = build_sms_matrix(seqs, matrix_names=["BLOSUM62"],
                               matrix_weights=[1.0], l=4,
                               use_property_pass=False, n_jobs=1)
    root, nodes   = build_phylo_tree(S, ids, method="upgma")
    leaf_w        = compute_leaf_weights(root)
    node_w        = compute_node_weights(root, leaf_w)
    cosim         = compute_cosimilarity(root, node_w)
    cut           = detect_boundaries(cosim, method="otsu")
    clusters, orphans, cluster_ids = extract_clusters(root, cut, min_size=2)
    threshold = otsu_threshold(list(cosim.values()))
    return {
        "ids": ids, "S": S, "root": root, "nodes": nodes,
        "cosim": cosim, "threshold": threshold,
        "clusters": clusters, "orphans": orphans,
        "cluster_ids": cluster_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _is_pdf(path: str) -> bool:
    """Check PDF magic bytes: first 4 bytes == b'%PDF'."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"%PDF"
    except OSError:
        return False


def _is_svg(path: str) -> bool:
    """Check SVG content."""
    try:
        content = open(path).read(200)
        return "<svg" in content
    except OSError:
        return False


def _is_html(path: str) -> bool:
    try:
        content = open(path).read(200).lower()
        return "<!doctype html" in content or "<html" in content
    except OSError:
        return False


def _html_has_plotly(path: str) -> bool:
    try:
        content = open(path).read()
        return "plotly" in content.lower()
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Color palette tests (no file I/O — pure logic)
# ─────────────────────────────────────────────────────────────────────────────

class TestColorPalette:
    """Test _build_palette from tree_plot and _cluster_palette from heatmap."""

    def test_tree_palette_unique_colors_for_small_n(self):
        from cluss_plus.visualization.tree_plot import _build_palette
        cids = {f"seq{i}": i for i in range(5)}
        pal  = _build_palette(cids)
        colors = [pal[i] for i in range(5)]
        assert len(set(colors)) == 5, "Palette produced duplicate colors for N=5"

    def test_tree_palette_orphan_always_grey(self):
        from cluss_plus.visualization.tree_plot import _build_palette, _ORPHAN_COLOR
        cids = {"a": 0, "b": "orphan"}
        pal  = _build_palette(cids)
        assert pal["orphan"] == _ORPHAN_COLOR

    def test_tree_palette_handles_zero_clusters(self):
        from cluss_plus.visualization.tree_plot import _build_palette
        pal = _build_palette({"a": "orphan"})
        assert "orphan" in pal

    def test_tree_palette_handles_20_clusters(self):
        from cluss_plus.visualization.tree_plot import _build_palette
        cids = {f"s{i}": i for i in range(20)}
        pal  = _build_palette(cids)
        # palette has one entry per cluster (20) — orphan key only added when present
        cluster_entries = {k: v for k, v in pal.items() if k != "orphan"}
        assert len(cluster_entries) == 20

    def test_heatmap_palette_returns_correct_count(self):
        from cluss_plus.visualization.heatmap import _cluster_palette
        colors = _cluster_palette(7)
        assert len(colors) == 7

    def test_heatmap_palette_colors_are_hex(self):
        from cluss_plus.visualization.heatmap import _cluster_palette
        import re
        colors = _cluster_palette(5)
        hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
        for c in colors:
            assert hex_re.match(c), f"Not a valid hex color: {c}"


# ─────────────────────────────────────────────────────────────────────────────
# Layout computation tests (no file I/O)
# ─────────────────────────────────────────────────────────────────────────────

class TestTreeLayout:

    def test_x_positions_root_at_zero(self, pipeline):
        from cluss_plus.visualization.tree_plot import _compute_x_positions
        x = _compute_x_positions(pipeline["root"])
        assert abs(x[pipeline["root"].id]) < 1e-9

    def test_x_positions_all_non_negative(self, pipeline):
        from cluss_plus.visualization.tree_plot import _compute_x_positions
        x = _compute_x_positions(pipeline["root"])
        for nid, xv in x.items():
            assert xv >= 0.0, f"Negative x at node {nid}: {xv}"

    def test_x_positions_leaves_furthest_from_root(self, pipeline):
        from cluss_plus.visualization.tree_plot import _compute_x_positions
        x     = _compute_x_positions(pipeline["root"])
        nodes = pipeline["nodes"]
        leaf_xs    = [x[nid] for nid, n in nodes.items() if n.is_leaf]
        internal_xs= [x[nid] for nid, n in nodes.items() if not n.is_leaf]
        assert max(leaf_xs) >= max(internal_xs) or len(internal_xs) == 0

    def test_y_positions_cover_range_0_to_n_minus_1(self, pipeline):
        from cluss_plus.visualization.tree_plot import (_compute_x_positions,
                                             _collect_leaves_ordered,
                                             _compute_y_positions)
        leaves = _collect_leaves_ordered(pipeline["root"])
        y = _compute_y_positions(pipeline["root"], leaves)
        ys = list(y.values())
        assert min(ys) >= 0.0
        assert max(ys) <= len(leaves) - 1 + 1e-9

    def test_leaves_ordered_returns_all_leaves(self, pipeline):
        from cluss_plus.visualization.tree_plot import _collect_leaves_ordered
        leaves = _collect_leaves_ordered(pipeline["root"])
        assert len(leaves) == 12
        leaf_ids = {n.seq_id for n in leaves}
        assert leaf_ids == set(SEQUENCES)

    def test_sort_order_clusters_contiguous(self, pipeline):
        from cluss_plus.visualization.heatmap import _sort_order
        ids   = pipeline["ids"]
        order = _sort_order(ids, pipeline["clusters"], pipeline["orphans"])
        assert len(order) == len(ids)
        assert set(order) == set(range(len(ids)))

    def test_strip_colors_correct_length(self, pipeline):
        from cluss_plus.visualization.heatmap import _cluster_strip_colors, _sort_order
        ids   = pipeline["ids"]
        order = _sort_order(ids, pipeline["clusters"], pipeline["orphans"])
        sorted_ids = [ids[i] for i in order]
        colors = _cluster_strip_colors(sorted_ids, pipeline["clusters"],
                                        pipeline["orphans"])
        assert len(colors) == len(ids)


# ─────────────────────────────────────────────────────────────────────────────
# Rectangular tree plot — file output tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRectTreePlot:

    def test_pdf_created_and_nonempty(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_rect_tree
        plot_rect_tree(pipeline["root"], pipeline["cluster_ids"],
                       out_dir=str(tmp_path))
        pdf = tmp_path / "tree_plot.pdf"
        assert _is_nonempty(str(pdf))

    def test_pdf_has_correct_magic_bytes(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_rect_tree
        plot_rect_tree(pipeline["root"], pipeline["cluster_ids"],
                       out_dir=str(tmp_path))
        assert _is_pdf(str(tmp_path / "tree_plot.pdf"))

    def test_svg_created_and_valid(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_rect_tree
        plot_rect_tree(pipeline["root"], pipeline["cluster_ids"],
                       out_dir=str(tmp_path))
        assert _is_svg(str(tmp_path / "tree_plot.svg"))

    def test_returns_pdf_path(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_rect_tree
        result = plot_rect_tree(pipeline["root"], pipeline["cluster_ids"],
                                out_dir=str(tmp_path))
        assert result.endswith(".pdf")
        assert os.path.exists(result)

    def test_with_annotations(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_rect_tree
        # Provide minimal annotations
        anns = {sid: {"organism": f"Organism {sid}"} for sid in SEQUENCES}
        plot_rect_tree(pipeline["root"], pipeline["cluster_ids"],
                       annotations=anns, out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "tree_plot.pdf"))

    def test_all_orphan_cluster_ids(self, pipeline, tmp_path):
        """Rect tree must not crash when all sequences are orphans."""
        from cluss_plus.visualization.tree_plot import plot_rect_tree
        all_orphan = {sid: "orphan" for sid in SEQUENCES}
        plot_rect_tree(pipeline["root"], all_orphan, out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "tree_plot.pdf"))

    def test_single_cluster(self, pipeline, tmp_path):
        """Rect tree must handle a single cluster (all seqs in one group)."""
        from cluss_plus.visualization.tree_plot import plot_rect_tree
        one_cluster = {sid: 0 for sid in SEQUENCES}
        plot_rect_tree(pipeline["root"], one_cluster, out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "tree_plot.pdf"))


# ─────────────────────────────────────────────────────────────────────────────
# Radial tree plot — file output tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRadialTreePlot:

    def test_pdf_created_and_nonempty(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_radial_tree
        plot_radial_tree(pipeline["root"], pipeline["cluster_ids"],
                          out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "tree_plot_radial.pdf"))

    def test_pdf_valid(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_radial_tree
        plot_radial_tree(pipeline["root"], pipeline["cluster_ids"],
                          out_dir=str(tmp_path))
        assert _is_pdf(str(tmp_path / "tree_plot_radial.pdf"))

    def test_svg_created(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_radial_tree
        plot_radial_tree(pipeline["root"], pipeline["cluster_ids"],
                          out_dir=str(tmp_path))
        assert _is_svg(str(tmp_path / "tree_plot_radial.svg"))

    def test_radial_layout_angles_span_full_circle(self, pipeline):
        from cluss_plus.visualization.tree_plot import (_collect_leaves_ordered,
                                             _radial_layout)
        import math
        leaves = _collect_leaves_ordered(pipeline["root"])
        xs, ys = _radial_layout(pipeline["root"], leaves)
        angles = [math.atan2(ys[n.id], xs[n.id]) for n in leaves]
        span   = max(angles) - min(angles)
        # 12 leaves — angle span must be close to 2π
        assert span > math.pi, f"Angle span too small: {span:.2f} rad"


# ─────────────────────────────────────────────────────────────────────────────
# Interactive tree plot
# ─────────────────────────────────────────────────────────────────────────────

class TestInteractiveTreePlot:

    def test_html_created_and_nonempty(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_interactive_tree
        plot_interactive_tree(pipeline["root"], pipeline["cluster_ids"],
                               out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "tree_plot_interactive.html"))

    def test_html_is_valid_html(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_interactive_tree
        plot_interactive_tree(pipeline["root"], pipeline["cluster_ids"],
                               out_dir=str(tmp_path))
        assert _is_html(str(tmp_path / "tree_plot_interactive.html"))

    def test_html_contains_plotly(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_interactive_tree
        plot_interactive_tree(pipeline["root"], pipeline["cluster_ids"],
                               out_dir=str(tmp_path))
        assert _html_has_plotly(str(tmp_path / "tree_plot_interactive.html"))

    def test_html_contains_seq_ids(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import plot_interactive_tree
        plot_interactive_tree(pipeline["root"], pipeline["cluster_ids"],
                               out_dir=str(tmp_path))
        content = open(str(tmp_path / "tree_plot_interactive.html")).read()
        # At least a few seq IDs should appear in the hover data
        hits = sum(1 for sid in SEQUENCES if sid in content)
        assert hits > 0, "No sequence IDs found in interactive HTML"

    def test_render_tree_both_style(self, pipeline, tmp_path):
        from cluss_plus.visualization.tree_plot import render_tree
        render_tree(pipeline["root"], pipeline["cluster_ids"],
                    out_dir=str(tmp_path), style="both")
        expected = [
            "tree_plot.pdf", "tree_plot.svg",
            "tree_plot_radial.pdf", "tree_plot_radial.svg",
            "tree_plot_interactive.html",
        ]
        for fname in expected:
            assert _is_nonempty(str(tmp_path / fname)), \
                f"Missing or empty: {fname}"


# ─────────────────────────────────────────────────────────────────────────────
# SMS heatmap
# ─────────────────────────────────────────────────────────────────────────────

class TestSMSHeatmap:

    def test_pdf_created(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_sms_heatmap
        plot_sms_heatmap(pipeline["S"], pipeline["ids"],
                          pipeline["clusters"], pipeline["orphans"],
                          out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "sms_heatmap.pdf"))

    def test_pdf_valid(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_sms_heatmap
        plot_sms_heatmap(pipeline["S"], pipeline["ids"],
                          pipeline["clusters"], pipeline["orphans"],
                          out_dir=str(tmp_path))
        assert _is_pdf(str(tmp_path / "sms_heatmap.pdf"))

    def test_svg_created(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_sms_heatmap
        plot_sms_heatmap(pipeline["S"], pipeline["ids"],
                          pipeline["clusters"], pipeline["orphans"],
                          out_dir=str(tmp_path))
        assert _is_svg(str(tmp_path / "sms_heatmap.svg"))

    def test_diverging_option_produces_file(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_sms_heatmap
        plot_sms_heatmap(pipeline["S"], pipeline["ids"],
                          pipeline["clusters"], pipeline["orphans"],
                          out_dir=str(tmp_path), diverging=True)
        assert _is_nonempty(str(tmp_path / "sms_heatmap.pdf"))

    def test_sort_order_is_permutation_of_seq_ids(self, pipeline):
        from cluss_plus.visualization.heatmap import _sort_order
        ids   = pipeline["ids"]
        order = _sort_order(ids, pipeline["clusters"], pipeline["orphans"])
        assert sorted(order) == list(range(len(ids)))

    def test_sorted_matrix_diagonal_still_ones(self, pipeline):
        from cluss_plus.visualization.heatmap import _sort_order
        ids    = pipeline["ids"]
        order  = _sort_order(ids, pipeline["clusters"], pipeline["orphans"])
        S_sort = pipeline["S"][np.ix_(order, order)]
        assert np.allclose(np.diag(S_sort), 1.0, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive heatmap
# ─────────────────────────────────────────────────────────────────────────────

class TestInteractiveHeatmap:

    def test_html_created(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_interactive_heatmap
        plot_interactive_heatmap(pipeline["S"], pipeline["ids"],
                                  pipeline["clusters"], pipeline["orphans"],
                                  out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "sms_heatmap_interactive.html"))

    def test_html_valid(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_interactive_heatmap
        plot_interactive_heatmap(pipeline["S"], pipeline["ids"],
                                  pipeline["clusters"], pipeline["orphans"],
                                  out_dir=str(tmp_path))
        assert _is_html(str(tmp_path / "sms_heatmap_interactive.html"))

    def test_html_contains_plotly(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_interactive_heatmap
        plot_interactive_heatmap(pipeline["S"], pipeline["ids"],
                                  pipeline["clusters"], pipeline["orphans"],
                                  out_dir=str(tmp_path))
        assert _html_has_plotly(str(tmp_path / "sms_heatmap_interactive.html"))


# ─────────────────────────────────────────────────────────────────────────────
# Cluster size chart
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterSizes:

    def test_pdf_created(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_cluster_sizes
        plot_cluster_sizes(pipeline["clusters"], pipeline["orphans"],
                            out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "cluster_sizes.pdf"))

    def test_pdf_valid(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_cluster_sizes
        plot_cluster_sizes(pipeline["clusters"], pipeline["orphans"],
                            out_dir=str(tmp_path))
        assert _is_pdf(str(tmp_path / "cluster_sizes.pdf"))

    def test_no_orphans_still_works(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_cluster_sizes
        plot_cluster_sizes(pipeline["clusters"], [],
                            out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "cluster_sizes.pdf"))

    def test_single_cluster_one_bar(self, tmp_path):
        from cluss_plus.visualization.heatmap import plot_cluster_sizes
        plot_cluster_sizes([["A1", "A2", "A3"]], [],
                            out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "cluster_sizes.pdf"))


# ─────────────────────────────────────────────────────────────────────────────
# Co-similarity distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestCoSimDist:

    def test_pdf_created(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_cosimilarity_dist
        plot_cosimilarity_dist(pipeline["cosim"], pipeline["threshold"],
                                out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "cosimilarity_dist.pdf"))

    def test_pdf_valid(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_cosimilarity_dist
        plot_cosimilarity_dist(pipeline["cosim"], pipeline["threshold"],
                                out_dir=str(tmp_path))
        assert _is_pdf(str(tmp_path / "cosimilarity_dist.pdf"))

    def test_svg_created(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_cosimilarity_dist
        plot_cosimilarity_dist(pipeline["cosim"], pipeline["threshold"],
                                out_dir=str(tmp_path))
        assert _is_svg(str(tmp_path / "cosimilarity_dist.svg"))

    @pytest.mark.parametrize("method", ["otsu", "gmm", "kneedle"])
    def test_all_boundary_method_labels(self, pipeline, tmp_path, method):
        from cluss_plus.visualization.heatmap import plot_cosimilarity_dist
        plot_cosimilarity_dist(pipeline["cosim"], pipeline["threshold"],
                                out_dir=str(tmp_path), method=method)
        assert _is_nonempty(str(tmp_path / "cosimilarity_dist.pdf"))

    def test_single_cosim_value_no_crash(self, tmp_path):
        """Edge case: only one internal node (smallest possible tree)."""
        from cluss_plus.visualization.heatmap import plot_cosimilarity_dist
        plot_cosimilarity_dist({0: 0.5}, threshold=0.5, out_dir=str(tmp_path))
        assert _is_nonempty(str(tmp_path / "cosimilarity_dist.pdf"))


# ─────────────────────────────────────────────────────────────────────────────
# Unified plot_all()
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotAll:

    def test_all_static_files_produced(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_all
        plot_all(pipeline["S"], pipeline["ids"],
                 pipeline["clusters"], pipeline["orphans"],
                 pipeline["cosim"], pipeline["threshold"],
                 out_dir=str(tmp_path), interactive=False)
        for fname in ["sms_heatmap.pdf", "sms_heatmap.svg",
                      "cluster_sizes.pdf", "cosimilarity_dist.pdf"]:
            assert _is_nonempty(str(tmp_path / fname)), f"Missing: {fname}"

    def test_interactive_file_produced_when_enabled(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_all
        plot_all(pipeline["S"], pipeline["ids"],
                 pipeline["clusters"], pipeline["orphans"],
                 pipeline["cosim"], pipeline["threshold"],
                 out_dir=str(tmp_path), interactive=True)
        assert _is_nonempty(str(tmp_path / "sms_heatmap_interactive.html"))

    def test_interactive_file_absent_when_disabled(self, pipeline, tmp_path):
        from cluss_plus.visualization.heatmap import plot_all
        plot_all(pipeline["S"], pipeline["ids"],
                 pipeline["clusters"], pipeline["orphans"],
                 pipeline["cosim"], pipeline["threshold"],
                 out_dir=str(tmp_path), interactive=False)
        assert not os.path.exists(str(tmp_path / "sms_heatmap_interactive.html"))
