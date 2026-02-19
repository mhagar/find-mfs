# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**find-mfs** is a Python package for finding molecular formula candidates from accurate mass values in mass spectrometry. It implements Böcker & Lipták's algorithm for mass decomposition using Extended Residue Tables (ERT), with chemical validation (octet rule, RDBE filtering) and isotope envelope matching.

## Build & Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_core.py

# Run a specific test
pytest tests/test_core.py::TestFormulaFinder::test_find_novobiocin
```

No linter or formatter is configured.

## Architecture

### Pipeline Flow

`FormulaFinder` (entry point) → `MassDecomposer` (candidate generation) → `FormulaValidator` (chemical filtering) → `FormulaSearchResults` (output container)

### Key Modules

- **`find_mfs/__init__.py`** — Public API exports and `find_chnops()` convenience function (uses a module-level singleton `FormulaFinder` for reuse)
- **`core/finder.py`** — `FormulaFinder` orchestrates the full pipeline: mass/charge/adduct correction, decomposition, validation, optional isotope matching
- **`core/decomposer.py`** — `MassDecomposer` implements the Böcker & Lipták ERT algorithm; uses pre-calculated tables from `data/` for CHNOPS and CHNOPS+halogens
- **`core/algorithms.py`** — Numba JIT-compiled decomposition functions (performance-critical inner loops)
- **`core/validator.py`** — `FormulaValidator` applies octet rule checks and RDBE (Ring/Double Bond Equivalents) filtering
- **`core/results.py`** — `FormulaSearchResults` container with iteration, slicing, pandas DataFrame export, and post-hoc filtering
- **`isotopes/`** — Isotope envelope simulation (via IsoSpecPy) and pattern matching against observed data
- **`utils/filtering.py`** — Octet rule and RDBE calculation logic
- **`utils/formulae.py`** — Formula constraint string parsing (e.g., `'C*H*N*O*P0S2'` → element bounds dict)

### Key Types

- `FormulaCandidate` — holds a `molmass.Formula` with computed error (ppm/Da), RDBE, and optional isotope match results
- `FormulaSearchResults` — list-like container of candidates with query metadata

### Dependencies

Core: `molmass`, `numpy`, `numba`, `IsoSpecPy`, `scipy`. Dev: `pytest`.
