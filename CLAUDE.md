# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**find-mfs** is a Python package for finding molecular formula candidates from accurate mass values in mass spectrometry. It implements Böcker & Lipták's algorithm for mass decomposition using Extended Residue Tables (ERT), with chemical validation (octet rule, RDBE filtering) and isotope envelope matching via IsoSpecPy.

## Build & Development Commands

```bash
# Install in development mode (builds Cython extensions)
pip install -e ".[dev]"

# Rebuild after modifying .pyx files
pip install -e ".[dev]"

# Run all tests (148 tests)
pytest

# Run a single test file
pytest tests/test_core.py

# Run a specific test
pytest tests/test_core.py::TestFormulaFinder::test_find_novobiocin

# Generate Cython annotation report (identify Python interaction overhead)
cython -a find_mfs/core/_algorithms.pyx
```

No linter or formatter is configured. The project uses `uv` for package management but standard `pip install -e .` also works.

## Architecture

### Pipeline Flow

```
FormulaFinder.find_formulae()
    ├─ Parse adduct, element constraints, prepare filter params
    ├─ MassDecomposer.decompose_and_score()  →  raw numpy arrays (counts, masses, errors, rdbe)
    │    └─ Cython _decompose_core() — recursive backtracking with on-the-fly RDBE/octet/isotope prefiltering
    ├─ Optional: score_isotope_batch() — batch IsoSpec C++ scoring via Cython dlopen bridge
    └─ Decision: lazy vs eager path
         ├─ Lazy → _LazyBackend wraps raw arrays, materializes FormulaCandidate on access
         └─ Eager → materializes all candidates, applies remaining validation
```

### Cython Extensions

Three `.pyx` files compiled via `setup.py` with `-O3 -ffast-math`:

- **`core/_algorithms.pyx`** — Decomposition kernel + fused scoring (exact mass, error, RDBE). The `_decompose_core` cdef function runs entirely nogil. Also has `decompose_and_score()` Python entry point that adds sorting. Header in `_algorithms.pxd`.
- **`core/_light_formula.pyx`** — `cdef class LightFormula` — C-struct-backed formula type with cached `.formula` and `.empirical` string builders. Duck-types as `molmass.Formula` for the properties used by the pipeline.
- **`isotopes/_isospec.pyx`** — Loads IsoSpecPy's C++ `.so` via `dlopen`/`dlsym` at module init, stores typed C function pointers. `_score_single_envelope()` runs entirely nogil. `score_isotope_batch()` loops over N candidates in a nogil block.

### Key Modules (Python)

- **`core/finder.py`** — `FormulaFinder` orchestrates the full pipeline. The lazy/eager decision depends on whether RDBE/octet filters were already applied in Cython (`can_prefilter`) or still need Python-level validation.
- **`core/decomposer.py`** — `MassDecomposer` builds the ERT and calls into Cython. Uses pre-calculated tables from `data/` for common element sets (CHNOPS, CHNOPS+halogens).
- **`core/results.py`** — `FormulaSearchResults` with dual backend: `_LazyBackend` (stores raw arrays, materializes on `__getitem__`) or eager list of `FormulaCandidate`. Filtering/sorting methods operate on arrays when lazy.
- **`core/validator.py`** — `FormulaValidator` applies RDBE range checks and octet rule.
- **`isotopes/envelope.py`** — High-level isotope matching API. `match_isotope_envelope()` and `score_isotope_batch()` both delegate to Cython C++ IsoSpecPy for scoring.
- **`isotopes/_isospec_bridge.py`** — `get_isotope_arrays()` converts element symbols to IsoSpec's flat array format (iso_numbers, flat_masses, flat_probs). Also has `M1_RATIOS`/`M2_DIRECT` dicts for approximate isotope prefiltering.
- **`utils/formulae.py`** — Parses constraint strings like `'C*H*N*O*P0S2'` into element bounds dicts.

### Key Types

- `FormulaCandidate` — dataclass holding a `LightFormula` (or `molmass.Formula`) with error (ppm/Da), RDBE, and optional `SingleEnvelopeMatchResult`
- `FormulaSearchResults` — list-like container with query metadata, `to_dataframe()`, filtering, and sorting
- `LightFormula` — Cython cdef class, duck-types as `molmass.Formula`. Constructed via `LightFormula.from_counts(symbols, counts, charge, mass)` which bypasses `__init__` for speed.

### Dependencies

Core: `molmass`, `numpy`, `IsoSpecPy`, `scipy`, `Cython>=3.0` (build). Dev: `pytest`, `pandas`.
