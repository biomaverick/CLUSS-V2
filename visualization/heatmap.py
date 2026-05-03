"""
visualization/heatmap.py
═════════════════════════
Similarity matrix heatmaps and cluster statistics charts.

Four outputs
────────────
1. plot_sms_heatmap()        — annotated N×N SMS similarity matrix heatmap,
                               sequences sorted by cluster membership so
                               within-cluster blocks appear on the diagonal.
2. plot_cluster_sizes()      — horizontal bar chart of cluster sizes.
3. plot_cosimilarity_dist()  — histogram of co-similarity values with the
                               Otsu/GMM/Kneedle threshold overlaid.
4. plot_all()                — renders all three figures in one call.

Design decisions
────────────────
- Sequences are re-ordered by cluster index so the block-diagonal structure
  is visible. Cluster boundaries are marked with white dividers.
- The color scale is sequential (white → deep blue) for similarity values,
  with a diverging option (white → blue for high, red for low) controlled
  by `diverging=True`.
- For large N (> 100) tick labels are hidden to prevent collision; a
  per-cluster colour strip replaces them along both axes.
- The interactive Plotly heatmap supports hover-text with seq_id pairs and
  their similarity score, useful for inspecting specific sequence pairs.
- All static figures are saved as PDF + SVG (vector, print-ready).
  Interactive figures are saved as self-contained HTML.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


# ─────────────────────────────────────────────────────────────────────────────
# Color helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "sms_blue",
    ["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    N=256,
)

_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "sms_div",
    ["#d73027", "#f46d43", "#fdae61", "#ffffff", "#abd9e9", "#4575b4", "#313695"],
    N=256,
)

_ORPHAN_COLOR = "#aaaaaa"


def _cluster_palette(n: int) -> list[str]:
    """Return n perceptually distinct hex colors."""
    if n <= 9:
        cmap = matplotlib.colormaps["Set1"].resampled(max(n, 1))
    else:
        cmap = matplotlib.colormaps["tab20"].resampled(n)
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def _sort_order(seq_ids: list[str],
                clusters: list[list[str]],
                orphans: list[str]) -> list[int]:
    """
    Return indices that sort seq_ids so clusters appear as contiguous blocks.
    Orphans go last.
    """
    id_to_idx = {sid: i for i, sid in enumerate(seq_ids)}
    order: list[int] = []
    for cluster in clusters:
        for sid in cluster:
            if sid in id_to_idx:
                order.append(id_to_idx[sid])
    for sid in orphans:
        if sid in id_to_idx:
            order.append(id_to_idx[sid])
    # Append any seq_ids not accounted for (shouldn't happen but safeguard)
    seen = set(order)
    for i in range(len(seq_ids)):
        if i not in seen:
            order.append(i)
    return order


def _cluster_strip_colors(seq_ids: list[str],
                           clusters: list[list[str]],
                           orphans: list[str]) -> list[str]:
    """
    Return a color string for each seq_id in seq_ids (after sorting).
    One color per cluster, orphans in grey.
    """
    palette  = _cluster_palette(len(clusters))
    cid_map: dict[str, str] = {}
    for i, cluster in enumerate(clusters):
        for sid in cluster:
            cid_map[sid] = palette[i]
    for sid in orphans:
        cid_map[sid] = _ORPHAN_COLOR
    return [cid_map.get(sid, _ORPHAN_COLOR) for sid in seq_ids]


# ─────────────────────────────────────────────────────────────────────────────
# 1. SMS similarity matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_sms_heatmap(S: np.ndarray,
                     seq_ids: list[str],
                     clusters: list[list[str]],
                     orphans: list[str],
                     out_dir: str = "output",
                     diverging: bool = False,
                     title: str = "SMS Pairwise Similarity Matrix",
                     dpi: int = 150) -> str:
    """
    Plot the N×N SMS similarity matrix as a heatmap.

    Sequences are re-ordered so each cluster forms a contiguous diagonal block.
    Cluster boundaries are marked with white dividing lines.
    A colour strip on both axes shows cluster membership.

    Parameters
    ----------
    S         : N×N float32 similarity matrix
    seq_ids   : list of sequence IDs (same order as S rows/columns)
    clusters  : list of lists (cluster membership)
    orphans   : list of orphan seq_ids
    out_dir   : output directory
    diverging : use diverging colormap centred on 0.5 (default: sequential)
    title     : figure title
    dpi       : raster DPI

    Returns
    -------
    Path to saved PDF.
    """
    os.makedirs(out_dir, exist_ok=True)

    N    = len(seq_ids)
    order = _sort_order(seq_ids, clusters, orphans)
    S_sorted = S[np.ix_(order, order)]
    sorted_ids = [seq_ids[i] for i in order]

    # Per-cluster boundary positions (cumulative cluster sizes)
    boundaries: list[int] = []
    pos = 0
    for cluster in clusters:
        pos += sum(1 for sid in cluster if sid in set(seq_ids))
        boundaries.append(pos)

    # Colors for the strip
    strip_colors = _cluster_strip_colors(sorted_ids, clusters, orphans)

    # Figure layout: main heatmap + two colour strips (x and y axis)
    strip_w = max(0.3, N * 0.02)   # strip width scales with N
    fig_size = max(8, N * 0.12 + 2)
    fig_size = min(fig_size, 28)   # cap at 28 inches

    fig = plt.figure(figsize=(fig_size + strip_w, fig_size + strip_w))
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[strip_w, fig_size],
        height_ratios=[fig_size, strip_w],
        hspace=0.01, wspace=0.01,
    )

    ax_heat  = fig.add_subplot(gs[0, 1])
    ax_ystrip = fig.add_subplot(gs[0, 0])
    ax_xstrip = fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[1, 0]).axis("off")   # corner spacer

    # ── Heatmap ────────────────────────────────────────────────────
    cmap = _DIVERGING_CMAP if diverging else _SEQUENTIAL_CMAP
    vmin, vmax = (0.0, 1.0)
    vcenter    = 0.5 if diverging else None

    norm = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax) \
           if diverging else mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax_heat.imshow(S_sorted, cmap=cmap, norm=norm,
                        aspect="equal", interpolation="nearest")

    # Cluster boundary lines
    for b in boundaries[:-1]:
        ax_heat.axhline(b - 0.5, color="white", lw=1.0, alpha=0.9)
        ax_heat.axvline(b - 0.5, color="white", lw=1.0, alpha=0.9)

    ax_heat.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # Tick labels (only show if N is small enough)
    if N <= 50:
        fsize = max(5, min(9, 400 / N))
        ax_heat.set_xticks(range(N))
        ax_heat.set_yticks(range(N))
        ax_heat.set_xticklabels(sorted_ids, rotation=90,
                                 ha="right", fontsize=fsize,
                                 fontfamily="monospace")
        ax_heat.set_yticklabels(sorted_ids, fontsize=fsize,
                                 fontfamily="monospace")
    else:
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.01)
    cbar.set_label("SMS similarity", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Y-axis colour strip ────────────────────────────────────────
    y_strip = np.array([mcolors.to_rgba(c) for c in strip_colors])
    ax_ystrip.imshow(y_strip.reshape(N, 1, 4), aspect="auto")
    for b in boundaries[:-1]:
        ax_ystrip.axhline(b - 0.5, color="white", lw=1.0)
    ax_ystrip.set_xticks([])
    ax_ystrip.set_yticks([])
    ax_ystrip.axis("off")

    # ── X-axis colour strip ────────────────────────────────────────
    x_strip = np.array([mcolors.to_rgba(c) for c in strip_colors])
    ax_xstrip.imshow(x_strip.reshape(1, N, 4), aspect="auto")
    for b in boundaries[:-1]:
        ax_xstrip.axvline(b - 0.5, color="white", lw=1.0)
    ax_xstrip.set_xticks([])
    ax_xstrip.set_yticks([])
    ax_xstrip.axis("off")

    # ── Legend ─────────────────────────────────────────────────────
    palette = _cluster_palette(len(clusters))
    handles = [
        mpatches.Patch(color=palette[i], label=f"Cluster {i}")
        for i in range(len(clusters))
    ]
    if orphans:
        handles.append(mpatches.Patch(color=_ORPHAN_COLOR, label="Orphan"))

    n_cols = max(1, min(4, len(handles) // 10 + 1))
    ax_heat.legend(handles=handles, loc="upper right",
                   fontsize=7.5, framealpha=0.85, ncol=n_cols,
                   title="Clusters", title_fontsize=7.5,
                   bbox_to_anchor=(1.0, 1.0))

    pdf_path = os.path.join(out_dir, "sms_heatmap.pdf")
    svg_path = os.path.join(out_dir, "sms_heatmap.svg")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  SMS heatmap → {pdf_path}")
    print(f"  SMS heatmap → {svg_path}")
    return pdf_path


# ─────────────────────────────────────────────────────────────────────────────
# 1b. Interactive Plotly heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_interactive_heatmap(S: np.ndarray,
                              seq_ids: list[str],
                              clusters: list[list[str]],
                              orphans: list[str],
                              out_dir: str = "output") -> str:
    """
    Interactive Plotly heatmap with hover-text for each cell.

    Hover shows: seq_id_i × seq_id_j and their similarity score.
    Saved as self-contained HTML.
    """
    import plotly.graph_objects as go

    os.makedirs(out_dir, exist_ok=True)

    order      = _sort_order(seq_ids, clusters, orphans)
    S_sorted   = S[np.ix_(order, order)]
    sorted_ids = [seq_ids[i] for i in order]
    N          = len(sorted_ids)

    # Build hover text matrix
    hover = [[f"{sorted_ids[r]} × {sorted_ids[c]}<br>SMS = {S_sorted[r,c]:.4f}"
              for c in range(N)] for r in range(N)]

    fig = go.Figure(go.Heatmap(
        z         = S_sorted.tolist(),
        x         = sorted_ids,
        y         = sorted_ids,
        text      = hover,
        hovertemplate = "%{text}<extra></extra>",
        colorscale = "Blues",
        zmin=0.0, zmax=1.0,
        colorbar  = dict(title="SMS similarity", thickness=14, len=0.7),
    ))

    # Cluster boundary shapes
    boundaries: list[int] = []
    pos = 0
    for cluster in clusters:
        pos += sum(1 for sid in cluster if sid in set(seq_ids))
        boundaries.append(pos - 0.5)

    shapes = []
    for b in boundaries[:-1]:
        shapes.append(dict(type="line", x0=b, x1=b, y0=-0.5, y1=N - 0.5,
                           line=dict(color="white", width=1.5)))
        shapes.append(dict(type="line", x0=-0.5, x1=N - 0.5, y0=b, y1=b,
                           line=dict(color="white", width=1.5)))

    tick_vals = list(range(N)) if N <= 60 else []
    tick_text = sorted_ids if N <= 60 else []

    fig.update_layout(
        title=dict(text="CLUSS+ SMS Similarity Matrix",
                   font=dict(size=15, family="Arial")),
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text,
                   tickangle=45, tickfont=dict(size=8, family="monospace"),
                   showgrid=False),
        yaxis=dict(tickvals=tick_vals, ticktext=tick_text,
                   tickfont=dict(size=8, family="monospace"),
                   showgrid=False, autorange="reversed"),
        shapes=shapes,
        width=900, height=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    html_path = os.path.join(out_dir, "sms_heatmap_interactive.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"  Interactive heatmap → {html_path}")
    return html_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cluster size bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_sizes(clusters: list[list[str]],
                       orphans: list[str],
                       out_dir: str = "output",
                       title: str = "Cluster Size Distribution") -> str:
    """
    Horizontal bar chart showing number of sequences per cluster.
    Orphans shown as a single bar in grey.

    Returns path to saved PDF.
    """
    os.makedirs(out_dir, exist_ok=True)

    palette  = _cluster_palette(len(clusters))
    labels   = [f"Cluster {i}" for i in range(len(clusters))]
    sizes    = [len(c) for c in clusters]
    colors   = palette[:]

    if orphans:
        labels.append("Orphans")
        sizes.append(len(orphans))
        colors.append(_ORPHAN_COLOR)

    n      = len(labels)
    fig_h  = max(4, n * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y_pos = range(n)
    bars  = ax.barh(list(y_pos), sizes, color=colors,
                    edgecolor="white", linewidth=0.6, height=0.7)

    # Value labels inside/outside bars
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_width() + max(sizes) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(size), va="center", ha="left", fontsize=9, fontweight="bold",
                color="#333333")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of sequences", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(0, max(sizes) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    ax.xaxis.set_tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.3, lw=0.6)

    plt.tight_layout()

    pdf_path = os.path.join(out_dir, "cluster_sizes.pdf")
    svg_path = os.path.join(out_dir, "cluster_sizes.svg")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cluster sizes → {pdf_path}")
    return pdf_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Co-similarity distribution histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_cosimilarity_dist(cosim: dict[int, float],
                            threshold: float,
                            out_dir: str = "output",
                            method: str = "otsu",
                            title: str = "Co-similarity Distribution") -> str:
    """
    Histogram of per-node co-similarity values with the detection threshold
    drawn as a vertical dashed line.

    The bimodal structure (low = cut-point nodes, high = compact subtrees)
    should be visible when the Otsu threshold is meaningful.

    Parameters
    ----------
    cosim     : dict[node_id -> co-similarity value]
    threshold : the threshold value used by detect_boundaries()
    out_dir   : output directory
    method    : 'otsu' | 'gmm' | 'kneedle' (label only)
    title     : figure title

    Returns
    -------
    Path to saved PDF.
    """
    os.makedirs(out_dir, exist_ok=True)

    values = np.array(list(cosim.values()), dtype=np.float64)
    n      = len(values)

    n_bins = max(15, min(60, int(math.sqrt(n) * 2)))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    low  = values[values <= threshold]
    high = values[values  > threshold]

    ax.hist(low,  bins=n_bins, color="#e74c3c", alpha=0.75,
            label=f"Cut nodes (≤ threshold)  n={len(low)}")
    ax.hist(high, bins=n_bins, color="#2980b9", alpha=0.75,
            label=f"Compact nodes (> threshold)  n={len(high)}")

    ax.axvline(threshold, color="#2c3e50", lw=2.0, ls="--",
               label=f"{method.capitalize()} threshold = {threshold:.4f}")

    ax.set_xlabel("Co-similarity value", fontsize=10)
    ax.set_ylabel("Number of internal nodes", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8.5, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3, lw=0.6)

    # Annotation: total nodes
    ax.text(0.98, 0.97, f"Total internal nodes: {n}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#555555")

    plt.tight_layout()

    pdf_path = os.path.join(out_dir, "cosimilarity_dist.pdf")
    svg_path = os.path.join(out_dir, "cosimilarity_dist.svg")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Co-similarity dist → {pdf_path}")
    return pdf_path


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(S: np.ndarray,
             seq_ids: list[str],
             clusters: list[list[str]],
             orphans: list[str],
             cosim: dict[int, float],
             threshold: float,
             out_dir: str = "output",
             boundary_method: str = "otsu",
             interactive: bool = True) -> None:
    """
    Render all heatmap visualizations in a single call.

    Produces:
      sms_heatmap.pdf / .svg
      cluster_sizes.pdf / .svg
      cosimilarity_dist.pdf / .svg
      sms_heatmap_interactive.html   (if interactive=True)

    Parameters
    ----------
    S                : N×N similarity matrix
    seq_ids          : sequence IDs (same order as S)
    clusters         : extracted clusters
    orphans          : orphan sequence IDs
    cosim            : co-similarity dict from cosimilarity.py
    threshold        : boundary detection threshold
    out_dir          : output directory
    boundary_method  : 'otsu' | 'gmm' | 'kneedle' (for dist plot label)
    interactive      : also write Plotly interactive HTML
    """
    print(f"\n  Rendering heatmap visualizations ...")
    plot_sms_heatmap(S, seq_ids, clusters, orphans, out_dir)
    plot_cluster_sizes(clusters, orphans, out_dir)
    plot_cosimilarity_dist(cosim, threshold, out_dir, method=boundary_method)
    if interactive:
        plot_interactive_heatmap(S, seq_ids, clusters, orphans, out_dir)
