# Contributing to CLUSS+

## Setup

```bash
git clone https://github.com/biomaverick/CLUSS-V2.git
cd CLUSS_V2
pip install -e ".[dev]"
```

## Running tests

```bash
# Fast tests only (excludes @pytest.mark.slow)
pytest tests/ -v

# Include slow / scalability tests
pytest tests/ -v -m slow
```

## Code style

- Max line length: 100 characters
- Run `flake8 cluss_plus/ --max-line-length=100 --ignore=E203,W503` before opening a PR
- All new modules must have a module-level docstring
- Use `logging` (never `print`) for all diagnostic output

## Logging conventions

Every module should configure its logger at the top:

```python
import logging
log = logging.getLogger(__name__)
```

Use `log.info()` for normal pipeline progress, `log.debug()` for verbose detail,
`log.warning()` for non-fatal issues, and `log.error()` for failures.

## Adding a new boundary method

1. Implement `your_method_threshold(values: list[float]) -> float` in
   `clustering/boundary_detector.py`
2. Add it to the `detect_boundaries()` dispatch dict
3. Add `"your_method"` to the `--boundary` choices in `main.py`
4. Add unit tests in `tests/test_pipeline_small.py`

## Adding a new substitution matrix

The SMS engine uses Biopython's `Bio.Align.substitution_matrices.load(name)`.
Any matrix name recognised by Biopython (e.g. `"PAM30"`, `"BLOSUM90"`) can be
passed via `--matrices`. No code changes needed.

## Pull request checklist

- [ ] All tests pass: `pytest tests/ -v -m "not slow"`
- [ ] Linting clean: `flake8 cluss_plus/`
- [ ] New behaviour is tested
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
