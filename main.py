"""
main.py
════════
CLUSS+ v2.0 — Alignment-Free Protein Sequence Clustering

Full pipeline entry point.

Quick start
───────────
  cluss+ --fasta proteins.fasta
  cluss+ --fasta proteins.fasta --mode hybrid --annotate
  cluss+ --fasta proteins.fasta --tree-method nj --boundary gmm --resume
  cluss+ --help
"""

import argparse
import logging
import os
import sys
import time
import json


log = logging.getLogger(__name__)


from preprocessing.fasta_parser    import parse_fasta, validate_sequences
from preprocessing.complexity_mask import (mask_all_sequences,
                                           mask_all_disordered)
from preprocessing.taxon_weights   import compute_taxon_weights

from similarity.sms_matrix  import (build_sms_matrix,
                                     build_esm2_matrix,
                                     build_hybrid_matrix)

from tree.phylo_tree         import build_phylo_tree

from clustering.cosimilarity       import (compute_leaf_weights,
                                           compute_node_weights,
                                           compute_cosimilarity)
from clustering.boundary_detector  import detect_boundaries
from clustering.cluster_extractor  import extract_clusters

from evaluation.q_measure          import (load_reference,
                                           compute_q_measure,
                                           compute_standard_metrics,
                                           save_metrics)
from evaluation.go_enrichment      import go_enrichment

from annotation.uniprot_fetcher    import fetch_all_annotations

from output.writer import (write_cluster_tsv,
                            write_cluster_fasta,
                            write_newick,
                            write_go_terms,
                            write_summary_json,
                            write_html_report,
                            save_checkpoint,
                            load_checkpoint)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "cluss+",
        description = "CLUSS+ v2.0 — Alignment-Free Protein Sequence Clustering",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--version", action="version", version="%(prog)s 2.0.0")  # Fix #9

    p.add_argument("--fasta", required=True,
                   help="Input FASTA file of protein sequences.")
    p.add_argument("--min-len", type=int, default=10,
                   help="Minimum sequence length; shorter sequences are dropped.")

    p.add_argument("--mode", choices=["sms", "esm2", "hybrid"], default="sms",
                   help="Similarity computation mode.")
    p.add_argument("--matrices", default="BLOSUM45,BLOSUM62,BLOSUM80",
                   help="Comma-separated substitution matrices for SMS blend.")
    p.add_argument("--matrix-weights", default="0.20,0.60,0.20",
                   help="Comma-separated blend weights (must sum to 1.0).")
    p.add_argument("--l", type=int, default=4,
                   help="Minimum motif length for SMS matching.")
    p.add_argument("--no-property-pass", action="store_true",
                   help="Disable Murphy property-group matching pass.")
    p.add_argument("--property-alphabet",
                   choices=["murphy8", "murphy10"], default="murphy8",
                   help="Reduced amino acid alphabet for property-group pass.")
    p.add_argument("--property-weight", type=float, default=0.3,
                   help="Blend weight for property-group pass (0-1).")
    p.add_argument("--esm-model", default="esm2_t6_8M_UR50D",
                   help="ESM-2 model name (used when --mode esm2 or hybrid).")
    p.add_argument("--sms-weight", type=float, default=0.5,
                   help="SMS fraction in hybrid mode (1 - sms-weight is ESM-2).")

    p.add_argument("--no-lcr-mask", action="store_true",
                   help="Disable low-complexity region (SEG) masking.")
    p.add_argument("--mask-idr", action="store_true",
                   help="Also mask intrinsically disordered regions via IUPred3 API.")
    p.add_argument("--lcr-k1", type=float, default=1.8,
                   help="SEG trigger entropy threshold (bits).")
    p.add_argument("--lcr-k2", type=float, default=2.5,
                   help="SEG extension entropy threshold (bits).")
    p.add_argument("--idr-threshold", type=float, default=0.5,
                   help="IUPred3 score cutoff for IDR masking.")

    p.add_argument("--tree-method", choices=["upgma", "nj"], default="upgma",
                   help="Phylogenetic tree construction method.")

    p.add_argument("--boundary", choices=["otsu", "gmm", "kneedle"],
                   default="otsu",
                   help="Co-similarity boundary detection method.")
    p.add_argument("--min-size", type=int, default=2,
                   help="Minimum cluster size; smaller groups become orphans.")

    p.add_argument("--annotate", action="store_true",
                   help="Fetch UniProt + InterPro annotations (requires internet).")
    p.add_argument("--no-interpro", action="store_true",
                   help="Skip InterPro domain fetching (faster annotation).")
    p.add_argument("--taxon-phylo", action="store_true",
                   help="Use NCBI phylogenetic taxon weighting (slower, more accurate).")

    p.add_argument("--reference",
                   help="Path to two-column TSV (seq_id TAB function_group) "
                        "for Q-measure and standard metrics.")

    p.add_argument("--out-dir", default="output",
                   help="Output directory.")
    p.add_argument("--split-fasta", action="store_true",
                   help="Write one FASTA file per cluster in output/per_cluster/.")
    p.add_argument("--no-fasta-out", action="store_true",
                   help="Skip writing FASTA outputs (saves time for large datasets).")

    p.add_argument("--plot", action="store_true",
                   help="Generate tree and heatmap visualizations.")
    p.add_argument("--plot-style", choices=["rect", "radial", "both"],
                   default="rect",
                   help="Tree visualization style.")
    p.add_argument("--no-interactive", action="store_true",
                   help="Skip interactive HTML plots (faster for large datasets).")

    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Number of parallel workers (-1 = all CPU cores).")
    p.add_argument("--chunk-size", type=int, default=5000,
                   help="Sequences per chunk for large-dataset matrix building. "
                        "Reduce if running out of memory.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from saved checkpoints. Skips completed stages.")

    p.add_argument("--log-level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   default="INFO",
                   help="Logging verbosity.")

    return p


def _tree_checkpoint_path(out_dir: str) -> str:
    return os.path.join(out_dir, "checkpoints", "tree.nwk")


def _save_tree(root, nodes, out_dir: str) -> None:
    """Serialise tree as Newick — ~100x smaller than pickle, human-readable."""
    path = _tree_checkpoint_path(out_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from tree.phylo_tree import to_newick as _to_newick
        newick_str = _to_newick(root)
        with open(path, "w") as fh:
            fh.write(newick_str)
            if not newick_str.endswith("\n"):
                fh.write("\n")
        log.info("Tree checkpoint saved (Newick): %s", path)
    except Exception as exc:
        log.warning("Could not save Newick checkpoint: %s — skipping.", exc)


def _load_tree(out_dir: str):
    """Load Newick checkpoint and reconstruct TreeNode graph."""
    path = _tree_checkpoint_path(out_dir)
    if not os.path.exists(path):
        return None, None
    try:
        from tree.phylo_tree import newick_to_treenodes
        root, nodes = newick_to_treenodes(path)
        log.info("Tree checkpoint loaded (Newick): %s", path)
        return root, nodes
    except Exception as exc:
        log.warning("Could not load Newick checkpoint: %s — rebuilding tree.", exc)
        return None, None


def run(args: argparse.Namespace) -> None:
    # Fix #8 (interim): raise recursion limit for moderate-N runs.
    # Fix #1 already converted the main traversals to iterative, but this
    # safety valve protects any future recursive paths added downstream.
    _required_depth = 200_000  # conservative ceiling for N up to ~100k
    if sys.getrecursionlimit() < _required_depth:
        sys.setrecursionlimit(_required_depth)
        log.warning("Raised recursion limit to %d.", _required_depth)

    t0 = time.time()

    log.info("=" * 60)
    log.info("CLUSS+  v2.0  --  Alignment-Free Protein Clustering")
    log.info("=" * 60)
    log.info("Input  : %s", args.fasta)
    log.info("Mode   : %s  |  Tree: %s  |  Boundary: %s",
             args.mode.upper(), args.tree_method.upper(), args.boundary.upper())
    log.info("Out dir: %s  |  Resume: %s", args.out_dir, args.resume)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── STAGE 1: Parse + validate ──────────────────────────────────
    log.info("[1/9] Parsing FASTA ...")
    sequences = parse_fasta(args.fasta)
    sequences = validate_sequences(sequences, min_len=args.min_len,
                                   output_dir=out_dir)
    if not sequences:
        log.error("No valid sequences remain after validation. Aborting.")
        sys.exit(1)

    seq_ids = list(sequences.keys())
    log.info("%d sequences ready for processing.", len(seq_ids))

    # ── STAGE 2: LCR / IDR masking ────────────────────────────────
    log.info("[2/9] Masking low-complexity regions ...")
    if not args.no_lcr_mask:
        masked_seqs = mask_all_sequences(sequences,
                                         k1=args.lcr_k1,
                                         k2=args.lcr_k2)
    else:
        masked_seqs = sequences
        log.info("LCR masking disabled (--no-lcr-mask).")

    if args.mask_idr:
        log.info("Applying IUPred3 IDR masking (this may take a while) ...")
        masked_seqs = mask_all_disordered(masked_seqs,
                                           threshold=args.idr_threshold,
                                           out_dir=out_dir)
    else:
        log.debug("IDR masking skipped (use --mask-idr to enable).")

    # ── STAGE 3: Annotations (optional) ───────────────────────────
    annotations: dict | None = None
    if args.annotate:
        log.info("[3/9] Fetching UniProt / InterPro annotations ...")
        annotations = fetch_all_annotations(
            seq_ids,
            fetch_interpro=(not args.no_interpro),
        )
    else:
        log.info("[3/9] Annotation skipped (use --annotate to enable).")

    taxon_weights: dict | None = None
    if annotations:
        log.info("Computing taxon diversity weights ...")
        taxon_weights = compute_taxon_weights(annotations,
                                               use_phylogenetic=args.taxon_phylo)

    # ── STAGE 4: Similarity matrix ─────────────────────────────────
    log.info("[4/9] Building similarity matrix ...")

    S_loaded = load_checkpoint("sms_matrix", out_dir) if args.resume else None

    if S_loaded is not None:
        S = S_loaded
        log.info("Resumed from checkpoint: %dx%d matrix", S.shape[0], S.shape[1])
    else:
        matrix_names   = [m.strip() for m in args.matrices.split(",")]
        matrix_weights = [float(w) for w in args.matrix_weights.split(",")]

        domains: dict | None = None
        if annotations:
            domains = {sid: ann.get("domains", [])
                       for sid, ann in annotations.items()
                       if ann.get("domains")}
            if not domains:
                domains = None

        sms_kwargs = dict(
            matrix_names      = matrix_names,
            matrix_weights    = matrix_weights,
            l                 = args.l,
            use_property_pass = not args.no_property_pass,
            property_weight   = args.property_weight,
            property_alphabet = args.property_alphabet,
            domains           = domains,
            n_jobs            = args.n_jobs,
            chunk_size        = args.chunk_size,
            out_dir           = out_dir,
        )

        if args.mode == "sms":
            seq_ids, S = build_sms_matrix(masked_seqs, **sms_kwargs)
        elif args.mode == "esm2":
            seq_ids, S = build_esm2_matrix(masked_seqs,
                                            model_name=args.esm_model)
        else:  # hybrid
            seq_ids, S = build_hybrid_matrix(
                masked_seqs,
                sms_weight = args.sms_weight,
                esm_weight = 1.0 - args.sms_weight,
                sms_kwargs = sms_kwargs,
                esm_model  = args.esm_model,
            )

        save_checkpoint("sms_matrix", S, out_dir)

    # ── STAGE 5: Phylogenetic tree ─────────────────────────────────
    log.info("[5/9] Building phylogenetic tree ...")
    root, nodes = _load_tree(out_dir) if args.resume else (None, None)

    if root is None:
        root, nodes = build_phylo_tree(S, seq_ids, method=args.tree_method)
        _save_tree(root, nodes, out_dir)

    # ── STAGE 6: Co-similarity ─────────────────────────────────────
    log.info("[6/9] Computing co-similarity ...")
    cosim_ckpt = load_checkpoint("cosimilarity", out_dir) if args.resume else None

    if cosim_ckpt is not None:
        cosim = {int(k): float(v) for k, v in cosim_ckpt.items()}
        log.info("Resumed from checkpoint (%d internal nodes).", len(cosim))
    else:
        leaf_weights = compute_leaf_weights(root)
        node_weights = compute_node_weights(root, leaf_weights)
        cosim        = compute_cosimilarity(root, node_weights)
        save_checkpoint("cosimilarity",
                        {str(k): v for k, v in cosim.items()},
                        out_dir)

    log.info("Co-similarity computed for %d internal nodes.", len(cosim))

    # ── STAGE 7: Boundary detection + cluster extraction ──────────
    log.info("[7/9] Detecting boundaries and extracting clusters ...")
    cut_nodes = detect_boundaries(cosim, method=args.boundary)
    clusters, orphans, cluster_ids = extract_clusters(
        root, cut_nodes, min_size=args.min_size
    )

    # ── STAGE 8: GO enrichment (optional) ─────────────────────────
    enriched_go: dict | None = None
    if args.annotate and annotations:
        log.info("[8/9] GO term enrichment ...")
        enriched_go = go_enrichment(clusters, annotations)
        write_go_terms(enriched_go, out_dir)
    else:
        log.info("[8/9] GO enrichment skipped (requires --annotate).")

    # ── STAGE 9: Metrics + output ──────────────────────────────────
    log.info("[9/9] Computing metrics and writing output ...")
    runtime = round(time.time() - t0, 2)

    metrics: dict = {"runtime_seconds": runtime}
    Q = None

    if args.reference:
        reference = load_reference(args.reference)
        Q         = compute_q_measure(clusters, orphans, reference, taxon_weights)
        std       = compute_standard_metrics(clusters, orphans, reference, S, seq_ids)
        metrics.update({"Q_measure": round(Q, 4)})
        metrics.update(std)
        save_metrics(Q, clusters, orphans, runtime,
                     output_dir = out_dir,
                     standard_metrics = std)
        log.info("Q-measure: %.2f%%", Q)
    else:
        log.info("No reference provided -- Q-measure skipped (use --reference).")
        path = os.path.join(out_dir, "metrics.json")
        with open(path, "w") as fh:
            json.dump({
                "n_clusters": len(clusters),
                "n_orphans":  len(orphans),
                "n_total":    sum(len(c) for c in clusters) + len(orphans),
                "cluster_sizes": sorted([len(c) for c in clusters], reverse=True),
                "runtime_seconds": runtime,
            }, fh, indent=2)

    run_meta = {
        "fasta":             args.fasta,
        "mode":              args.mode,
        "tree_method":       args.tree_method,
        "boundary":          args.boundary,
        "min_motif_l":       args.l,
        "matrices":          args.matrices,
        "property_pass":     not args.no_property_pass,
        "property_alphabet": args.property_alphabet,
        "lcr_masking":       not args.no_lcr_mask,
        "idr_masking":       args.mask_idr,
        "annotated":         args.annotate,
        "n_jobs":            args.n_jobs,
    }

    write_newick(root, out_dir)
    write_cluster_tsv(clusters, orphans, annotations, out_dir)

    if not args.no_fasta_out:
        write_cluster_fasta(sequences, clusters, orphans, out_dir,
                            split=args.split_fasta)

    write_summary_json(run_meta, clusters, orphans, metrics, out_dir)
    write_html_report(
        clusters    = clusters,
        orphans     = orphans,
        metrics     = metrics,
        annotations = annotations,
        enriched_go = enriched_go,
        run_meta    = run_meta,
        out_dir     = out_dir,
    )

    if args.plot:
        from clustering.boundary_detector import (otsu_threshold, gmm_threshold,
                                                   kneedle_threshold)
        from visualization.tree_plot import render_tree
        from visualization.heatmap   import plot_all as plot_heatmaps

        threshold_fn = {"otsu": otsu_threshold,
                        "gmm":  gmm_threshold,
                        "kneedle": kneedle_threshold}.get(args.boundary, otsu_threshold)
        vis_threshold = threshold_fn(list(cosim.values()))

        render_tree(root, cluster_ids, annotations,
                    out_dir=out_dir, style=args.plot_style)
        plot_heatmaps(S, seq_ids, clusters, orphans, cosim,
                      vis_threshold, out_dir=out_dir,
                      boundary_method=args.boundary,
                      interactive=not args.no_interactive)

    log.info("=" * 60)
    log.info("Done in %.1fs  |  %d clusters  |  %d orphans",
             runtime, len(clusters), len(orphans))
    if Q is not None:
        log.info("Q-measure: %.2f%%", Q)
    log.info("Results -> %s/", os.path.abspath(out_dir))
    log.info("=" * 60)


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    run(args)


if __name__ == "__main__":
    main()
