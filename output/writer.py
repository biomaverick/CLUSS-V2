"""
output/writer.py
═════════════════
CLUSS+ v2.0 — Output serialisation helpers.

All eight functions write files into *out_dir* and create it (and any
sub-directories) if it does not yet exist.  None of them raise on missing
optional inputs — ``None`` is handled gracefully everywhere.

Functions
─────────
write_cluster_tsv      → clusters.tsv  +  orphans.tsv
write_cluster_fasta    → clusters.fasta  [+  per_cluster/*.fasta]
write_newick           → tree.nwk
write_go_terms         → go_terms.tsv
write_summary_json     → summary.json
write_html_report      → report.html
save_checkpoint        → checkpoints/<name>.json  (or .pkl for non-JSON)
load_checkpoint        → deserialised object  |  None
"""

import json
import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure(path: str) -> str:
    """Create parent directories for *path* and return *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _out(out_dir: str, filename: str) -> str:
    """Return full path inside *out_dir*, creating it if needed."""
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TSV cluster tables
# ─────────────────────────────────────────────────────────────────────────────

def write_cluster_tsv(
    clusters: list[list[str]],
    orphans: list[str],
    annotations: "dict | None",
    out_dir: str,
) -> None:
    """
    Write ``clusters.tsv`` and ``orphans.tsv`` into *out_dir*.

    clusters.tsv columns
    ────────────────────
    cluster_id  seq_id  size  function  organism  (annotation columns optional)

    orphans.tsv columns
    ───────────────────
    seq_id  function  organism

    Parameters
    ----------
    clusters    : list of clusters, each a list of sequence IDs
    orphans     : list of orphan sequence IDs
    annotations : {seq_id: {function, organism, ...}} or None
    out_dir     : output directory
    """
    ann = annotations or {}

    # ── clusters.tsv ──────────────────────────────────────────────
    cluster_path = _out(out_dir, "clusters.tsv")
    with open(cluster_path, "w") as fh:
        fh.write("cluster_id\tseq_id\tcluster_size\tfunction\torganism\n")
        for cid, members in enumerate(clusters, start=1):
            for seq_id in members:
                a = ann.get(seq_id, {})
                func = a.get("function", "")
                org  = a.get("organism", "")
                fh.write(f"{cid}\t{seq_id}\t{len(members)}\t{func}\t{org}\n")

    log.info("Written: %s  (%d clusters)", cluster_path, len(clusters))

    # ── orphans.tsv ───────────────────────────────────────────────
    orphan_path = _out(out_dir, "orphans.tsv")
    with open(orphan_path, "w") as fh:
        fh.write("seq_id\tfunction\torganism\n")
        for seq_id in orphans:
            a = ann.get(seq_id, {})
            fh.write(f"{seq_id}\t{a.get('function','')}\t{a.get('organism','')}\n")

    log.info("Written: %s  (%d orphans)", orphan_path, len(orphans))


# ─────────────────────────────────────────────────────────────────────────────
# 2. FASTA cluster output
# ─────────────────────────────────────────────────────────────────────────────

def write_cluster_fasta(
    sequences: "dict[str, str]",
    clusters: "list[list[str]]",
    orphans: "list[str]",
    out_dir: str,
    split: bool = False,
) -> None:
    """
    Write sequences labelled by cluster into FASTA files.

    Always writes ``clusters.fasta`` (all clustered sequences) and
    ``orphans.fasta``.  When *split* is True, also writes one file per
    cluster into ``per_cluster/cluster_<N>.fasta``.

    Parameters
    ----------
    sequences : {seq_id: sequence_string}
    clusters  : list of clusters, each a list of sequence IDs
    orphans   : list of orphan sequence IDs
    out_dir   : output directory
    split     : if True write per-cluster FASTA files as well
    """
    # ── combined clusters.fasta ───────────────────────────────────
    combined_path = _out(out_dir, "clusters.fasta")
    with open(combined_path, "w") as fh:
        for cid, members in enumerate(clusters, start=1):
            for seq_id in members:
                seq = sequences.get(seq_id, "")
                fh.write(f">{seq_id}  cluster={cid}\n{seq}\n")

    log.info("Written: %s", combined_path)

    # ── orphans.fasta ─────────────────────────────────────────────
    orphan_path = _out(out_dir, "orphans.fasta")
    with open(orphan_path, "w") as fh:
        for seq_id in orphans:
            seq = sequences.get(seq_id, "")
            fh.write(f">{seq_id}  orphan\n{seq}\n")

    log.info("Written: %s  (%d orphans)", orphan_path, len(orphans))

    # ── per_cluster/*.fasta ───────────────────────────────────────
    if split:
        per_dir = os.path.join(out_dir, "per_cluster")
        os.makedirs(per_dir, exist_ok=True)
        for cid, members in enumerate(clusters, start=1):
            path = os.path.join(per_dir, f"cluster_{cid:04d}.fasta")
            with open(path, "w") as fh:
                for seq_id in members:
                    seq = sequences.get(seq_id, "")
                    fh.write(f">{seq_id}\n{seq}\n")
        log.info("Written per-cluster FASTA files to %s/", per_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Newick tree
# ─────────────────────────────────────────────────────────────────────────────

def write_newick(root: Any, out_dir: str) -> None:
    """
    Write the phylogenetic tree as a Newick string to ``tree.nwk``.

    Delegates to ``root.to_newick()`` (defined on TreeNode).  If the
    root object does not expose ``to_newick`` the call is silently skipped
    to avoid blocking the rest of the pipeline.

    Parameters
    ----------
    root    : TreeNode (root of the phylogenetic tree)
    out_dir : output directory
    """
    if root is None:
        log.warning("write_newick: root is None — skipping.")
        return

    to_newick_fn = getattr(root, "to_newick", None)
    if to_newick_fn is None:
        # fall back: try importing from tree module
        try:
            from tree.phylo_tree import to_newick as _to_newick
            newick_str = _to_newick(root)
        except Exception as exc:
            log.warning("write_newick: cannot serialise tree (%s) — skipping.", exc)
            return
    else:
        newick_str = to_newick_fn()

    path = _out(out_dir, "tree.nwk")
    with open(path, "w") as fh:
        fh.write(newick_str)
        if not newick_str.endswith("\n"):
            fh.write("\n")

    log.info("Written: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GO enrichment table
# ─────────────────────────────────────────────────────────────────────────────

def write_go_terms(
    enriched_go: "dict | None",
    out_dir: str,
) -> None:
    """
    Write ``go_terms.tsv`` from the GO enrichment dictionary.

    Expected dict structure
    ───────────────────────
    {cluster_id: [{"go_id": str, "term": str, "p_value": float, ...}, ...]}

    Silently writes an empty file when *enriched_go* is None or empty.

    Parameters
    ----------
    enriched_go : GO enrichment results or None
    out_dir     : output directory
    """
    path = _out(out_dir, "go_terms.tsv")
    with open(path, "w") as fh:
        fh.write("cluster_id\tgo_id\tterm\tp_value\tengenes\n")
        if enriched_go:
            for cid, terms in enriched_go.items():
                for entry in (terms or []):
                    go_id   = entry.get("go_id",   "")
                    term    = entry.get("term",     "")
                    pval    = entry.get("p_value",  "")
                    engenes = entry.get("genes",    "")
                    if isinstance(engenes, (list, tuple)):
                        engenes = ",".join(str(g) for g in engenes)
                    fh.write(f"{cid}\t{go_id}\t{term}\t{pval}\t{engenes}\n")

    log.info("Written: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary JSON
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_json(
    run_meta: dict,
    clusters: "list[list[str]]",
    orphans: "list[str]",
    metrics: dict,
    out_dir: str,
) -> None:
    """
    Write a machine-readable ``summary.json`` run record.

    Parameters
    ----------
    run_meta : dict with pipeline parameters (fasta, mode, etc.)
    clusters : list of clusters
    orphans  : list of orphan sequence IDs
    metrics  : dict with Q-measure, runtime, etc.
    out_dir  : output directory
    """
    summary = {
        "cluss_plus_version": "2.0.0",
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_parameters": run_meta,
        "results": {
            "n_clusters":    len(clusters),
            "n_orphans":     len(orphans),
            "n_clustered":   sum(len(c) for c in clusters),
            "cluster_sizes": sorted([len(c) for c in clusters], reverse=True),
        },
        "metrics": metrics,
    }

    path = _out(out_dir, "summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    log.info("Written: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. HTML report
# ─────────────────────────────────────────────────────────────────────────────

def write_html_report(
    clusters: "list[list[str]]",
    orphans: "list[str]",
    metrics: dict,
    annotations: "dict | None",
    enriched_go: "dict | None",
    run_meta: dict,
    out_dir: str,
) -> None:
    """
    Write a self-contained HTML summary report (``report.html``).

    The report includes:
      - Run parameters table
      - Metrics summary
      - Cluster size distribution bar chart (inline SVG)
      - Per-cluster accordion with member sequence IDs and annotations
      - GO term enrichment table (if available)

    All styling is inline CSS — no external dependencies.

    Parameters
    ----------
    clusters    : list of clusters, each a list of sequence IDs
    orphans     : list of orphan sequence IDs
    metrics     : dict with Q-measure, runtime, etc.
    annotations : {seq_id: {...}} or None
    enriched_go : GO enrichment dict or None
    run_meta    : pipeline parameter dict
    out_dir     : output directory
    """
    ann = annotations or {}
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Cluster bar chart (inline SVG) ────────────────────────────
    sizes = sorted([len(c) for c in clusters], reverse=True)
    max_size = max(sizes, default=1)
    bar_h = 12
    bar_gap = 4
    chart_w = 400
    n_bars = min(len(sizes), 30)   # show at most 30 bars
    chart_h = n_bars * (bar_h + bar_gap) + 20

    bars_svg = []
    for k, sz in enumerate(sizes[:n_bars]):
        bar_w = max(2, int(sz / max_size * (chart_w - 80)))
        y = k * (bar_h + bar_gap) + 10
        bars_svg.append(
            f'<rect x="60" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'fill="#4a90d9" rx="2"/>'
            f'<text x="55" y="{y + bar_h - 2}" text-anchor="end" '
            f'font-size="9" fill="#555">{k+1}</text>'
            f'<text x="{60 + bar_w + 4}" y="{y + bar_h - 2}" '
            f'font-size="9" fill="#555">{sz}</text>'
        )
    bars_svg_str = "\n".join(bars_svg)
    chart_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{chart_w}" height="{chart_h}">'
        f'<text x="200" y="8" text-anchor="middle" font-size="10" '
        f'font-weight="bold" fill="#333">Cluster sizes (top {n_bars})</text>'
        f"{bars_svg_str}</svg>"
    )

    # ── Run parameters table ──────────────────────────────────────
    param_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in run_meta.items()
    )

    # ── Metrics table ─────────────────────────────────────────────
    metric_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in metrics.items()
    )

    # ── Cluster accordion ─────────────────────────────────────────
    cluster_sections = []
    for cid, members in enumerate(clusters, start=1):
        rows = []
        for seq_id in members:
            a    = ann.get(seq_id, {})
            func = a.get("function", "—")
            org  = a.get("organism", "—")
            rows.append(f"<tr><td>{seq_id}</td><td>{func}</td><td>{org}</td></tr>")
        table = (
            "<table><thead><tr><th>Seq ID</th><th>Function</th>"
            f"<th>Organism</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
        cluster_sections.append(
            f'<details><summary>Cluster {cid} &nbsp;'
            f'<span class="badge">{len(members)}</span></summary>{table}</details>'
        )
    clusters_html = "\n".join(cluster_sections)

    # ── GO terms table ────────────────────────────────────────────
    go_html = ""
    if enriched_go:
        go_rows = []
        for cid, terms in enriched_go.items():
            for entry in (terms or []):
                go_rows.append(
                    f"<tr><td>{cid}</td><td>{entry.get('go_id','')}</td>"
                    f"<td>{entry.get('term','')}</td>"
                    f"<td>{entry.get('p_value',''):.3e}</td></tr>"
                    if isinstance(entry.get("p_value"), float)
                    else
                    f"<tr><td>{cid}</td><td>{entry.get('go_id','')}</td>"
                    f"<td>{entry.get('term','')}</td>"
                    f"<td>{entry.get('p_value','')}</td></tr>"
                )
        go_html = (
            "<h2>GO Term Enrichment</h2>"
            "<table><thead><tr><th>Cluster</th><th>GO ID</th>"
            "<th>Term</th><th>p-value</th></tr></thead>"
            f"<tbody>{''.join(go_rows)}</tbody></table>"
        )

    # ── Orphan list ───────────────────────────────────────────────
    orphan_rows = "".join(
        f"<tr><td>{s}</td><td>{ann.get(s,{}).get('function','—')}</td></tr>"
        for s in orphans
    )
    orphan_html = (
        f'<details><summary>Orphan sequences &nbsp;'
        f'<span class="badge">{len(orphans)}</span></summary>'
        f'<table><thead><tr><th>Seq ID</th><th>Function</th></tr></thead>'
        f'<tbody>{orphan_rows}</tbody></table></details>'
    ) if orphans else ""

    # ── Assemble full page ────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CLUSS+ Report</title>
<style>
  body {{font-family: system-ui, sans-serif; margin: 2rem; color: #222; max-width: 960px;}}
  h1 {{color: #1a5276;}} h2 {{color: #1f618d; margin-top: 2rem;}}
  table {{border-collapse: collapse; width: 100%; margin: 0.5rem 0;}}
  th, td {{border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.88rem;}}
  th {{background: #eaf2fb;}}
  tr:nth-child(even) {{background: #f8f9fa;}}
  details {{margin: 0.4rem 0; border: 1px solid #d5dbdb; border-radius: 4px;}}
  summary {{padding: 8px 12px; cursor: pointer; background: #eaf2fb;
            font-weight: 600; border-radius: 4px;}}
  summary:hover {{background: #d6eaf8;}}
  .badge {{background: #2980b9; color: #fff; border-radius: 10px;
           padding: 1px 8px; font-size: 0.8rem; font-weight: normal;}}
  .grid {{display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;}}
  footer {{margin-top: 3rem; font-size: 0.8rem; color: #888;}}
</style>
</head>
<body>
<h1>CLUSS+ Clustering Report</h1>
<p>Generated: {timestamp}</p>

<div class="grid">
  <div>
    <h2>Run Parameters</h2>
    <table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>
    <tbody>{param_rows}</tbody></table>
  </div>
  <div>
    <h2>Metrics</h2>
    <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{metric_rows}</tbody></table>
  </div>
</div>

<h2>Cluster Size Distribution</h2>
{chart_svg}

<h2>Clusters ({len(clusters)} total, {sum(len(c) for c in clusters)} sequences)</h2>
{clusters_html}

{orphan_html}

{go_html}

<footer>
  CLUSS+ v2.0 &mdash; Kelil et al. (2007) BMC Bioinformatics 8:286 &mdash;
  <a href="https://github.com/your-org/cluss-plus">GitHub</a>
</footer>
</body>
</html>
"""

    path = _out(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)

    log.info("Written: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 7 & 8. Checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(name: str, obj: Any, out_dir: str) -> None:
    """
    Persist *obj* to ``checkpoints/<name>.json`` (JSON-serialisable objects)
    or ``checkpoints/<name>.pkl`` (fallback for non-JSON types such as numpy
    arrays).

    Parameters
    ----------
    name    : checkpoint identifier (no extension needed)
    obj     : Python object to serialise
    out_dir : output root directory
    """
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Try JSON first (portable, human-readable)
    json_path = os.path.join(ckpt_dir, f"{name}.json")
    try:
        serialised = json.dumps(obj, default=_json_default)
        with open(json_path, "w") as fh:
            fh.write(serialised)
        log.debug("Checkpoint saved (JSON): %s", json_path)
        return
    except (TypeError, ValueError):
        pass

    # Fallback to pickle for numpy arrays and other non-JSON types
    pkl_path = os.path.join(ckpt_dir, f"{name}.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.debug("Checkpoint saved (pickle): %s", pkl_path)


def load_checkpoint(name: str, out_dir: str) -> Any:
    """
    Load a checkpoint saved by :func:`save_checkpoint`.

    Returns the deserialised object, or ``None`` if no checkpoint exists
    for *name* (the stage will then run from scratch).

    Parameters
    ----------
    name    : checkpoint identifier (no extension)
    out_dir : output root directory

    Returns
    -------
    Deserialised object or ``None``
    """
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    json_path = os.path.join(ckpt_dir, f"{name}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path) as fh:
                obj = json.load(fh)
            log.debug("Checkpoint loaded (JSON): %s", json_path)
            return obj
        except Exception as exc:
            log.warning("Failed to load JSON checkpoint %s: %s", json_path, exc)

    pkl_path = os.path.join(ckpt_dir, f"{name}.pkl")
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as fh:
                obj = pickle.load(fh)
            log.debug("Checkpoint loaded (pickle): %s", pkl_path)
            return obj
        except Exception as exc:
            log.warning("Failed to load pickle checkpoint %s: %s", pkl_path, exc)

    return None


def _json_default(obj: Any) -> Any:
    """JSON serialiser for types not handled by the stdlib encoder."""
    # numpy scalars / arrays
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
