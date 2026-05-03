# Changelog

## [2.0.1] — 2026-05-03 (GitHub-ready patch)

### Critical fixes
- `clustering/cosimilarity.py`: Converted all three recursive traversals
  (`compute_leaf_weights`, `compute_node_weights`, `compute_cosimilarity`)
  to explicit stack-based iteration — eliminates `RecursionError` on N > ~500
  with unbalanced UPGMA trees (Python default recursion limit = 1 000;
  tree depth can exceed 50 000 for large N).
- `clustering/cluster_extractor.py`: Converted `collect_leaves` and
  `_traverse` to iterative DFS — same crash class as above.
- `pyproject.toml`: Added `[tool.setuptools.packages.find]` directive;
  without it `pip install .` silently installs an empty package.

### Should-fix improvements
- `clustering/boundary_detector.py`: Replaced `print()` with `log.info()`
  so output respects `--log-level` and HPC (SLURM) log capture.
- `clustering/cluster_extractor.py`: Same `print()` → `log.info()` fix.
- `SECURITY.md`: Replaced unreachable 2007 paper-author email with
  GitHub private security advisory link.
- `__init__.py`: Split `__author__` (maintainer) from `__paper__`
  (original paper authors) to avoid misattribution.
- Added `CITATION.cff` — enables GitHub "Cite this repository" button
  and correct DOI attribution.
- `README.md`: Added Scalability section with runtime estimates and
  recommendations for N > 10 000.

### Nice-to-have improvements
- `main.py`: Added `--version` flag (`cluss+ --version` now works).
- `main.py`: Added interim `sys.setrecursionlimit(200_000)` guard in
  `run()` as a safety valve for any future recursive additions.
- `similarity/sms_engine.py`: Added Numba JIT warm-up call at import
  time so first-run compilation delay is visible before the progress bar.

## [2.0.0] — 2026-05-03

### Bug fixes
- `sms_engine.py`: Fixed `find_seeds()` recording wrong X seed position (off by l-1)
- `sms_matrix.py`: Fixed `compute_s_max` returning per-residue average instead of total weight
- `phylo_tree.py`: Fixed `TreeNode.depth` never being assigned (RF boundary always returned 0)
- `q_measure.py`: Clamped Q-measure to [0, 100] to prevent negative values for large orphan counts
- `boundary_detector.py`: Doubled Otsu candidate thresholds from 100 to 200

### Infrastructure
- Added `pyproject.toml` with all dependencies and entry point
- Added `LICENSE` (MIT)
- Added `.gitignore` (excludes `output/`, checkpoints, model caches)
- Added `.github/workflows/ci.yml` for GitHub Actions (Python 3.9 / 3.10 / 3.11)
- Added `CHANGELOG.md` and `CONTRIBUTING.md`
- Added `install.sh` convenience installer
- Replaced all `print()` calls with structured `logging` (controlled via `--log-level`)
- Added `--log-level` CLI flag (DEBUG / INFO / WARNING / ERROR)
- Added `--chunk-size` CLI flag for large-dataset matrix building
- Added `np.memmap`-backed matrix storage for datasets > 10k sequences
- Mocked UniProt / IUPred3 HTTP calls in test suite (CI-safe)
- Added `__version__` to `cluss_plus/__init__.py`
- Removed committed runtime artifact `output/dropped.tsv`

### Biological upgrades
- Replaced Taylor 6-class alphabet with Murphy8 / Murphy10 reduced alphabets
- Added two-threshold SEG masking (k1=1.8, k2=2.5 bits)
- Added IUPred3 IDR masking (optional, `--mask-idr`)
- Added BLOSUM45/62/80 multi-matrix blend
- Added Neighbor Joining tree method (`--tree-method nj`)
- Added GMM and Kneedle boundary detection (`--boundary gmm|kneedle`)
- Added ESM-2 hybrid mode (`--mode hybrid|esm2`)
- Added GO semantic redundancy reduction
- Added ARI, NMI, and Silhouette metrics

## [1.0.0] — original Kelil et al. (2007) implementation
