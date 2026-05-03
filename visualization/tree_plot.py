"""
visualization/tree_plot.py
═══════════════════════════
Radial (circular) and rectangular cladogram visualizations of the
CLUSS+ phylogenetic tree, color-coded by cluster membership.

Two renderers
─────────────
1. plot_radial_tree()  — circular layout; best for ≤ 200 sequences
2. plot_rect_tree()    — rectangular (top-down) cladogram; scales to
                         larger datasets without label collision

Both produce a publication-quality figure saved to out_dir/tree_plot.pdf
and an interactive Plotly HTML copy to out_dir/tree_plot_interactive.html.

Design decisions
────────────────
- Layout is computed from scratch in pure NumPy — no networkx/ETE3 dependency.
  This keeps the install footprint small and avoids version conflicts.
- Leaf labels are rendered at the tip of each branch. Label font size is
  scaled automatically to fit N leaves without overlap.
- Branch lengths from the TreeNode are used directly, so the visual
  distances reflect the evolutionary distance estimates in the tree.
- The cluster palette uses ColorBrewer Qualitative Set1 extended with
  tab20 for datasets with >9 clusters. Orphans are always grey.
- A cluster legend is placed in the upper-right corner.
- The Plotly interactive version includes hover text (seq_id, organism
  if annotations are present, cluster ID, branch length).

Usage (programmatic)
────────────────────
  from visualization.tree_plot import plot_radial_tree, plot_rect_tree

  plot_radial_tree(root, cluster_ids, annotations, out_dir="output")
  plot_rect_tree  (root, cluster_ids, annotations, out_dir="output")

Usage (CLI integration — called from main.py --plot-tree)
──────────────────────────────────────────────────────────
  from visualization.tree_plot import render_tree
  render_tree(root, cluster_ids, annotations, out_dir, style="radial")
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import to_hex

from tree.phylo_tree import TreeNode

# ─────────────────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────────────────

_ORPHAN_COLOR = "#aaaaaa"

def _build_palette(cluster_ids: dict[str, int | str]) -> dict[int | str, str]:
    """
    Assign a distinct hex color to each cluster index.
    Uses ColorBrewer Set1 (9 colors) + matplotlib tab20 for larger sets.
    Orphans always get _ORPHAN_COLOR.
    """
    unique = sorted(
        {v for v in cluster_ids.values() if v != "orphan"},
        key=lambda x: int(x) if isinstance(x, int) else 0
    )
    n = len(unique)

    if n <= 9:
        # ColorBrewer Set1 — perceptually distinct, print-safe
        cmap = matplotlib.colormaps["Set1"].resampled(max(n, 1))
    else:
        cmap = matplotlib.colormaps["tab20"].resampled(n)

    palette: dict[int | str, str] = {
        cid: to_hex(cmap(i / max(n - 1, 1))) for i, cid in enumerate(unique)
    }
    palette["orphan"] = _ORPHAN_COLOR
    return palette


# ─────────────────────────────────────────────────────────────────────────────
# Layout computation — shared by both renderers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_leaves_ordered(root: TreeNode) -> list[TreeNode]:
    """DFS in-order traversal to collect leaves (preserves tree order)."""
    leaves = []
    stack  = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            leaves.append(node)
        else:
            for child in reversed(node.children):
                stack.append(child)
    return leaves


def _compute_x_positions(root: TreeNode) -> dict[int, float]:
    """
    Compute x-position (= cumulative branch length from root) for every node.
    Root is at x=0. Leaves are at their total path length from root.
    """
    x: dict[int, float] = {}

    def _visit(node: TreeNode, parent_x: float) -> None:
        node_x = parent_x + node.branch_length
        x[node.id] = node_x
        for child in node.children:
            _visit(child, node_x)

    _visit(root, 0.0)
    return x


def _compute_y_positions(root: TreeNode,
                          leaves: list[TreeNode]) -> dict[int, float]:
    """
    Assign y-positions.
    Leaves are spaced evenly from 0 to N-1.
    Internal nodes are centered between their children's y-range.
    """
    leaf_y = {node.id: float(i) for i, node in enumerate(leaves)}
    y: dict[int, float] = {}

    def _visit(node: TreeNode) -> float:
        if node.is_leaf:
            y[node.id] = leaf_y[node.id]
            return y[node.id]
        child_ys = [_visit(c) for c in node.children]
        y[node.id] = (min(child_ys) + max(child_ys)) / 2.0
        return y[node.id]

    _visit(root)
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Rectangular cladogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_rect_tree(root: TreeNode,
                   cluster_ids: dict[str, int | str],
                   annotations: dict[str, dict] | None = None,
                   out_dir: str = "output",
                   title: str = "CLUSS+ Phylogenetic Tree",
                   dpi: int = 150) -> str:
    """
    Rectangular cladogram with horizontal branches, colored by cluster.

    Parameters
    ----------
    root        : root TreeNode
    cluster_ids : dict[seq_id -> cluster_index or 'orphan']
    annotations : optional dict[seq_id -> annotation_dict] for organism labels
    out_dir     : directory to write tree_plot.pdf and tree_plot.svg
    title       : figure title
    dpi         : raster DPI for PNG fallback

    Returns
    -------
    Path to the saved PDF.
    """
    os.makedirs(out_dir, exist_ok=True)

    palette  = _build_palette(cluster_ids)
    leaves   = _collect_leaves_ordered(root)
    N        = len(leaves)
    x_pos    = _compute_x_positions(root)
    y_pos    = _compute_y_positions(root, leaves)

    # Figure dimensions — scale height with leaf count
    fig_h = max(6, N * 0.28)
    fig_w = max(14, fig_h * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Draw branches ──────────────────────────────────────────────
    def _draw_node(node: TreeNode) -> None:
        if node.is_leaf:
            return
        x_par  = x_pos[node.id]
        y_par  = y_pos[node.id]

        # Vertical connector between children
        child_ys = [y_pos[c.id] for c in node.children]
        ax.plot([x_par, x_par], [min(child_ys), max(child_ys)],
                color="#555555", lw=0.9, zorder=1)

        for child in node.children:
            cid   = cluster_ids.get(child.seq_id, "orphan") if child.is_leaf else None
            color = palette.get(cid, _ORPHAN_COLOR) if cid is not None else "#555555"
            ax.plot([x_par, x_pos[child.id]], [y_pos[child.id], y_pos[child.id]],
                    color=color, lw=1.4, zorder=2, solid_capstyle="round")
            _draw_node(child)

    _draw_node(root)

    # ── Leaf labels ────────────────────────────────────────────────
    max_x     = max(x_pos.values()) if x_pos else 1.0
    label_pad = max_x * 0.015

    font_size = max(4.5, min(10, 200 / max(N, 1)))

    for leaf in leaves:
        cid   = cluster_ids.get(leaf.seq_id, "orphan")
        color = palette.get(cid, _ORPHAN_COLOR)

        # Primary label: seq_id
        label = leaf.seq_id
        # Optional organism suffix (if annotations available)
        if annotations and leaf.seq_id in annotations:
            org = annotations[leaf.seq_id].get("organism", "")
            if org and org != "Unknown":
                org_short = org.split()[0]  # genus only for brevity
                label = f"{leaf.seq_id}  [{org_short}]"

        ax.text(x_pos[leaf.id] + label_pad, y_pos[leaf.id],
                label, va="center", ha="left",
                fontsize=font_size, color=color, fontfamily="monospace")

        ax.plot(x_pos[leaf.id], y_pos[leaf.id], "o",
                color=color, ms=3.5, zorder=3)

    # ── Axis formatting ────────────────────────────────────────────
    ax.set_xlim(-max_x * 0.02, max_x * 1.45)
    ax.set_ylim(-1, N)
    ax.set_xlabel("Evolutionary distance  (1 − SMS similarity)",
                  fontsize=9, labelpad=6)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=8)

    # ── Legend ─────────────────────────────────────────────────────
    unique_cids = sorted(
        {v for v in cluster_ids.values() if v != "orphan"},
        key=lambda x: int(x) if isinstance(x, int) else 0
    )
    legend_handles = [
        mpatches.Patch(color=palette[cid], label=f"Cluster {cid}")
        for cid in unique_cids
    ]
    if any(v == "orphan" for v in cluster_ids.values()):
        legend_handles.append(
            mpatches.Patch(color=_ORPHAN_COLOR, label="Orphan")
        )

    n_cols = max(1, min(4, len(legend_handles) // 8 + 1))
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8, framealpha=0.85, ncol=n_cols,
              title="Clusters", title_fontsize=8)

    plt.tight_layout()

    pdf_path = os.path.join(out_dir, "tree_plot.pdf")
    svg_path = os.path.join(out_dir, "tree_plot.svg")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tree plot  → {pdf_path}")
    print(f"  Tree plot  → {svg_path}")
    return pdf_path


# ─────────────────────────────────────────────────────────────────────────────
# Radial (circular) cladogram
# ─────────────────────────────────────────────────────────────────────────────

def _radial_layout(root: TreeNode,
                   leaves: list[TreeNode]
                   ) -> tuple[dict[int, float], dict[int, float]]:
    """
    Compute (x, y) Cartesian coordinates for a radial tree layout.

    Each leaf is assigned an angle evenly spaced in [0, 2π).
    Internal nodes are placed at the mean angle of their descendants,
    at radius = cumulative branch length from root.

    Returns two dicts: {node.id -> x}, {node.id -> y}
    """
    N = len(leaves)
    leaf_angle = {node.id: 2.0 * math.pi * i / N
                  for i, node in enumerate(leaves)}

    radii: dict[int, float] = {}
    angles: dict[int, float] = {}

    def _visit(node: TreeNode, parent_radius: float) -> float:
        r = parent_radius + node.branch_length
        radii[node.id] = r
        if node.is_leaf:
            angles[node.id] = leaf_angle[node.id]
            return angles[node.id]
        child_angles = [_visit(c, r) for c in node.children]
        angles[node.id] = (min(child_angles) + max(child_angles)) / 2.0
        return angles[node.id]

    _visit(root, 0.0)

    xs = {nid: radii[nid] * math.cos(angles[nid]) for nid in radii}
    ys = {nid: radii[nid] * math.sin(angles[nid]) for nid in radii}
    return xs, ys


def plot_radial_tree(root: TreeNode,
                     cluster_ids: dict[str, int | str],
                     annotations: dict[str, dict] | None = None,
                     out_dir: str = "output",
                     title: str = "CLUSS+ Radial Phylogenetic Tree",
                     dpi: int = 150) -> str:
    """
    Circular/radial phylogenetic tree colored by cluster membership.

    Best for datasets up to ~200 sequences. Labels are placed outside
    the outermost ring; font size scales with N to prevent collision.

    Parameters
    ----------
    root        : root TreeNode
    cluster_ids : dict[seq_id -> cluster_index or 'orphan']
    annotations : optional organism labels
    out_dir     : output directory
    title       : figure title
    dpi         : raster resolution

    Returns
    -------
    Path to saved PDF.
    """
    os.makedirs(out_dir, exist_ok=True)

    palette = _build_palette(cluster_ids)
    leaves  = _collect_leaves_ordered(root)
    N       = len(leaves)

    xs, ys = _radial_layout(root, leaves)

    # Figure size — circular so use square canvas
    fig_size = max(10, N * 0.18 + 4)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    # ── Draw branches ──────────────────────────────────────────────
    def _draw_arc(x1: float, y1: float,
                  x2: float, y2: float,
                  color: str, lw: float) -> None:
        """Draw a straight line between two points (simplified arc)."""
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
                zorder=2, solid_capstyle="round")

    def _draw_node(node: TreeNode) -> None:
        if node.is_leaf:
            return
        for child in node.children:
            # Elbow: vertical segment from parent angle to child angle,
            # then radial segment from parent radius to child.
            # Approximate with two straight segments: parent→elbow→child.
            px, py = xs[node.id],  ys[node.id]
            cx, cy = xs[child.id], ys[child.id]

            # Elbow point: child's angle, parent's radius
            r_par  = math.hypot(px, py)
            if r_par < 1e-9:
                # Root is at center — draw directly
                _draw_arc(px, py, cx, cy, "#555555", 0.8)
                _draw_node(child)
                continue

            ang_child = math.atan2(cy, cx)
            ex = r_par * math.cos(ang_child)
            ey = r_par * math.sin(ang_child)

            # Circular arc from parent to elbow (approximate with line)
            _draw_arc(px, py, ex, ey, "#555555", 0.8)

            # Radial line from elbow to child (colored by cluster)
            if child.is_leaf:
                cid   = cluster_ids.get(child.seq_id, "orphan")
                color = palette.get(cid, _ORPHAN_COLOR)
            else:
                color = "#555555"
            _draw_arc(ex, ey, cx, cy, color, 1.3)
            _draw_node(child)

    _draw_node(root)

    # ── Leaf labels ────────────────────────────────────────────────
    max_r = max(math.hypot(x, y) for x, y in zip(xs.values(), ys.values()))
    if max_r < 1e-9:
        max_r = 1.0
    pad    = max_r * 0.06
    fsize  = max(4, min(9, 180 / max(N, 1)))

    for leaf in leaves:
        cid   = cluster_ids.get(leaf.seq_id, "orphan")
        color = palette.get(cid, _ORPHAN_COLOR)

        angle = math.atan2(ys[leaf.id], xs[leaf.id])
        lx    = (max_r + pad) * math.cos(angle)
        ly    = (max_r + pad) * math.sin(angle)

        # Rotate label so it reads outward
        deg = math.degrees(angle)
        if -90 <= deg <= 90:
            rot = deg
            ha  = "left"
        else:
            rot = deg + 180
            ha  = "right"

        label = leaf.seq_id
        if annotations and leaf.seq_id in annotations:
            org = annotations[leaf.seq_id].get("organism", "")
            if org and org != "Unknown":
                label = f"{leaf.seq_id} [{org.split()[0]}]"

        ax.text(lx, ly, label, ha=ha, va="center",
                rotation=rot, rotation_mode="anchor",
                fontsize=fsize, color=color, fontfamily="monospace")

        ax.plot(xs[leaf.id], ys[leaf.id], "o", color=color, ms=3, zorder=4)

    # ── Formatting ─────────────────────────────────────────────────
    margin = max_r * 1.55
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)

    # ── Legend ─────────────────────────────────────────────────────
    unique_cids = sorted(
        {v for v in cluster_ids.values() if v != "orphan"},
        key=lambda x: int(x) if isinstance(x, int) else 0
    )
    legend_handles = [
        mpatches.Patch(color=palette[cid], label=f"Cluster {cid}")
        for cid in unique_cids
    ]
    if any(v == "orphan" for v in cluster_ids.values()):
        legend_handles.append(mpatches.Patch(color=_ORPHAN_COLOR, label="Orphan"))

    n_cols = max(1, min(3, len(legend_handles) // 8 + 1))
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8.5, framealpha=0.88, ncol=n_cols,
              title="Clusters", title_fontsize=8.5,
              bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()

    pdf_path    = os.path.join(out_dir, "tree_plot_radial.pdf")
    svg_path    = os.path.join(out_dir, "tree_plot_radial.svg")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Radial tree → {pdf_path}")
    print(f"  Radial tree → {svg_path}")
    return pdf_path


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Plotly tree
# ─────────────────────────────────────────────────────────────────────────────

def plot_interactive_tree(root: TreeNode,
                           cluster_ids: dict[str, int | str],
                           annotations: dict[str, dict] | None = None,
                           out_dir: str = "output") -> str:
    """
    Export an interactive Plotly line-plot tree with hover labels.
    Saved to tree_plot_interactive.html (self-contained, opens in browser).

    Hover text shows: seq_id, organism (if available), cluster ID,
    and branch length.

    Parameters
    ----------
    root        : root TreeNode
    cluster_ids : dict[seq_id -> cluster_index or 'orphan']
    annotations : optional seq_id -> annotation_dict
    out_dir     : output directory

    Returns
    -------
    Path to the HTML file.
    """
    import plotly.graph_objects as go

    os.makedirs(out_dir, exist_ok=True)
    palette = _build_palette(cluster_ids)
    leaves  = _collect_leaves_ordered(root)
    x_pos   = _compute_x_positions(root)
    y_pos   = _compute_y_positions(root, leaves)

    edge_traces: list = []
    leaf_traces: dict[str, dict] = {}   # keyed by color hex

    def _collect(node: TreeNode) -> None:
        if node.is_leaf:
            return
        xp = x_pos[node.id]
        yp = y_pos[node.id]
        child_ys = [y_pos[c.id] for c in node.children]

        # Vertical connector
        edge_traces.append(
            go.Scatter(x=[xp, xp], y=[min(child_ys), max(child_ys)],
                       mode="lines",
                       line=dict(color="#888888", width=1.0),
                       hoverinfo="none", showlegend=False)
        )

        for child in node.children:
            if child.is_leaf:
                cid   = cluster_ids.get(child.seq_id, "orphan")
                color = palette.get(cid, _ORPHAN_COLOR)
                # Hover text
                org   = ""
                if annotations and child.seq_id in annotations:
                    org = annotations[child.seq_id].get("organism", "")
                hover = (f"<b>{child.seq_id}</b><br>"
                         f"Cluster: {cid}<br>"
                         f"Branch length: {child.branch_length:.4f}"
                         + (f"<br>Organism: {org}" if org and org != "Unknown" else ""))
            else:
                color = "#888888"
                hover = f"Internal node<br>Branch: {child.branch_length:.4f}"

            edge_traces.append(
                go.Scatter(x=[xp, x_pos[child.id]],
                           y=[yp, y_pos[child.id]],
                           mode="lines",
                           line=dict(color=color, width=1.5),
                           hoverinfo="none", showlegend=False)
            )

            if child.is_leaf:
                cid   = cluster_ids.get(child.seq_id, "orphan")
                color = palette.get(cid, _ORPHAN_COLOR)
                key   = str(cid)
                if key not in leaf_traces:
                    leaf_traces[key] = {"x": [], "y": [], "text": [],
                                        "color": color, "cid": cid}
                leaf_traces[key]["x"].append(x_pos[child.id])
                leaf_traces[key]["y"].append(y_pos[child.id])
                leaf_traces[key]["text"].append(hover)

            _collect(child)

    _collect(root)

    # Leaf scatter traces (one per cluster → shared legend)
    leaf_scatter = []
    for key, d in sorted(leaf_traces.items()):
        cid_label = f"Cluster {d['cid']}" if d["cid"] != "orphan" else "Orphan"
        leaf_scatter.append(
            go.Scatter(x=d["x"], y=d["y"],
                       mode="markers+text",
                       name=cid_label,
                       marker=dict(color=d["color"], size=7,
                                   line=dict(width=0.5, color="#ffffff")),
                       text=[t.split("<br>")[0].replace("<b>","").replace("</b>","")
                             for t in d["text"]],
                       textposition="middle right",
                       textfont=dict(size=9, family="monospace",
                                     color=d["color"]),
                       hovertemplate="%{customdata}<extra></extra>",
                       customdata=d["text"],
                       showlegend=True)
        )

    fig = go.Figure(data=edge_traces + leaf_scatter)
    fig.update_layout(
        title=dict(text="CLUSS+ Interactive Phylogenetic Tree",
                   font=dict(size=16, family="Arial")),
        xaxis=dict(title="Evolutionary distance (1 − SMS similarity)",
                   showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(title="Clusters", bordercolor="#dddddd",
                    borderwidth=1, bgcolor="rgba(255,255,255,0.9)"),
        hovermode="closest",
        margin=dict(l=40, r=40, t=60, b=50),
    )
    fig.update_xaxes(showline=True, linecolor="#cccccc")

    html_path = os.path.join(out_dir, "tree_plot_interactive.html")
    fig.write_html(html_path, include_plotlyjs="cdn",
                   config={"toImageButtonOptions": {"format": "svg"}})
    print(f"  Interactive tree → {html_path}")
    return html_path


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_tree(root: TreeNode,
                cluster_ids: dict[str, int | str],
                annotations: dict[str, dict] | None = None,
                out_dir: str = "output",
                style: str = "rect") -> None:
    """
    Render all tree visualizations.

    Parameters
    ----------
    root        : root TreeNode
    cluster_ids : dict[seq_id -> cluster_index or 'orphan']
    annotations : optional annotation dict for organism labels
    out_dir     : output directory
    style       : 'rect' (default) | 'radial' | 'both'
                  'both' renders rectangular + radial + interactive
    """
    N = sum(1 for n in _collect_leaves_ordered(root))
    print(f"\n  Rendering tree visualizations ({N} leaves, style={style}) ...")

    if style in ("rect", "both"):
        plot_rect_tree(root, cluster_ids, annotations, out_dir)

    if style in ("radial", "both") or (style == "rect" and N <= 200):
        if N <= 200:
            plot_radial_tree(root, cluster_ids, annotations, out_dir)
        else:
            print(f"  Radial tree skipped (N={N} > 200; use style='radial' to force)")

    # Interactive always rendered
    plot_interactive_tree(root, cluster_ids, annotations, out_dir)
