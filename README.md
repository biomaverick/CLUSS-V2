# CLUSS+ v2.0 — Alignment-Free Protein Sequence Clustering

Based on Kelil et al. (2007) *BMC Bioinformatics* 8:286, with all reviewed
bug fixes and biologically-motivated upgrades applied.

---

## Quick start

```bash
git clone https://github.com/yourorg/cluss-plus.git
cd cluss-plus
chmod +x install.sh && ./install.sh

# Minimal run
cluss+ --fasta proteins.fasta

# Full run: hybrid similarity, NJ tree, annotations, GO enrichment
cluss+ --fasta proteins.fasta \
       --mode hybrid --tree-method nj --boundary gmm \
       --annotate --reference ref_classification.tsv \
       --out-dir results/

# Resume an interrupted run
cluss+ --fasta proteins.fasta --resume
```

---

## Architecture

```
FASTA parse
  └─ validate (length, alphabet)
      └─ LCR masking (2-threshold SEG)     [--no-lcr-mask to skip]
          └─ IDR masking (IUPred3 API)     [--mask-idr to enable]
              └─ UniProt/InterPro annotation [--annotate]
                  └─ SMS similarity matrix  [--mode sms|esm2|hybrid]
                      └─ Phylogenetic tree  [--tree-method upgma|nj]
                          └─ Co-similarity  (Ward / Batagelj)
                              └─ Boundaries [--boundary otsu|gmm|kneedle]
                                  └─ Clusters + orphans
                                      └─ GO enrichment + metrics
                                          └─ Output files
```

---

## Outputs

| File | Description |
|------|-------------|
| `clusters.tsv` | Cluster assignment for every sequence |
| `orphans.tsv` | Sequences below `--min-size` threshold |
| `clusters.fasta` | FASTA with cluster IDs in headers |
| `tree.nwk` | Phylogenetic tree (Newick format) |
| `summary.json` | Full run metadata and metrics |
| `report.html` | Self-contained HTML report |
| `go_terms.tsv` | Enriched GO terms per cluster (if `--annotate`) |
| `metrics.json` | Q-measure, ARI, NMI, Silhouette |
| `checkpoints/` | Resume state for `--resume` |

---

## Key CLI flags

```
--fasta          Input FASTA (required)
--mode           sms | esm2 | hybrid            (default: sms)
--tree-method    upgma | nj                     (default: upgma)
--boundary       otsu | gmm | kneedle           (default: otsu)
--l              Minimum motif length           (default: 4)
--matrices       BLOSUM matrix blend            (default: BLOSUM45,BLOSUM62,BLOSUM80)
--annotate       Fetch UniProt + InterPro annotations
--mask-idr       Also mask IDRs via IUPred3 API
--reference      Reference TSV for Q-measure evaluation
--out-dir        Output directory               (default: output)
--resume         Resume from saved checkpoints
--n-jobs         Parallel workers               (default: -1 = all cores)
```

---

## Bug fixes over original paper implementation

| Module | Bug | Fix |
|--------|-----|-----|
| `sms_engine.py` | `find_seeds()` subtracted `(l-1)` from `run_start`, giving wrong X positions | `run_start` is already correct; removed erroneous subtraction |
| `sms_matrix.py` | `compute_s_max` returned per-residue average; short-pair scores clipped to 1.0 | Returns total weight; normalize all pairs by `len(longest)` |
| `phylo_tree.py` | `TreeNode.depth` never assigned; RF boundary feature always returned 0 | `assign_depths()` called after tree build |
| `q_measure.py` | Q-measure could go negative for large orphan counts | Clamped to `[0, 100]` |
| `boundary_detector.py` | Otsu used 100 candidate thresholds | Doubled to 200 for finer resolution |

---

## Biological upgrades

| Feature | Details |
|---------|---------|
| Murphy8/10 reduced alphabets | Replace incorrect Taylor 6-class grouping (Cys ≠ Ser/Thr) |
| Two-threshold SEG masking | Trigger k1=1.8 bits + extend k2=2.5 bits (original SEG paper) |
| IUPred3 IDR masking | Masks disordered regions by energy model, not entropy |
| BLOSUM45/62/80 blend | Captures substitutions at multiple evolutionary distances |
| Neighbor Joining tree | Correct for rate variation across lineages (no clock assumption) |
| GMM / Kneedle boundaries | Robust alternatives when co-similarity distribution is non-bimodal |
| ESM-2 hybrid mode | Protein language model for distant homologue detection |
| Domain-aware SMS | Up-weights residues inside InterPro globular domain boundaries |
| GO semantic reduction | Removes ancestor redundancy from enriched GO term lists |
| ARI + NMI + Silhouette | Standard external + internal clustering quality metrics |

---

## Scalability

The SMS pairwise matrix is **O(N²)** in both time and memory. Use the table
below to gauge expected runtime before launching large jobs.

| N (sequences) | Pairs   | Estimated wall time (32 cores, SMS mode) |
|---------------|---------|------------------------------------------|
| 500           | 125 k   | < 1 min                                  |
| 2 000         | 2 M     | ~5 min                                   |
| 10 000        | 50 M    | ~4 h                                     |
| 50 000        | 1.25 B  | ~4 days                                  |

> **Warning:** Running `cluss+ --fasta large.fasta` on more than ~10 000
> sequences without pre-filtering will queue a multi-day job.

**Recommended strategies for large N:**

- Pre-cluster at 90 % identity with [CD-HIT](https://github.com/weizhongli/cdhit),
  then run CLUSS+ per cluster.
- Use `--mode hybrid` with an ESM-2 GPU backend (much faster pairwise than
  SMS on CPU).
- Reduce `--chunk-size` if you hit memory limits during matrix construction.

---

## References

1. Kelil et al. (2007) CLUSS. *BMC Bioinformatics* 8:286
2. Murphy et al. (2000) Reduced alphabets. *Protein Engineering* 13:149
3. Wootton & Federhen (1993) SEG algorithm. *Comput Chem* 17:149
4. Saitou & Nei (1987) Neighbor Joining. *Mol Biol Evol* 4:406
5. Erdős et al. (2021) IUPred3. *Nucleic Acids Res* 49:W297
6. Lin et al. (2023) ESM-2. *Science* 379:1123
