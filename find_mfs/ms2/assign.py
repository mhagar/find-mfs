"""
Subformula assignment (i.e. for signals in an MS2 spectrum)

Given a candidate root formula + ion and an MS2 `SpectrumArray`,
 produces the `(subformula, intensity, ppm)` matches which are featurized for MistNet

Replaces MIST-CF's `chem_utils.assign_subforms` lattice enumeration with
constrained find-mfs queries (`max_counts=root`)

Note! **THIS ASSUMES THE MS2 SCAN HAS ALREADY BEEN DE-ISOTOPED!**

It assigns the nearest subformula per peak, independent per peak.

There used to be a second mode (`isotopes=True`) doing greedy 'backwards'
isotope handling per candidate root. I removed it for now, but will re-visit.
Strangely, it measured worse than the naive 'assign formula to every peak' approach

Unified, spectrum-level envelope handling belongs here in find-mfs (`find_mfs.spectra.envelopes`);
see mist-fmfs `docs/PROJECT_STATE.md` Sec 5b.i for the intended follow-up
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

import molmass

from find_mfs.spectra import SpectrumArray
from find_mfs import FormulaFinder

from find_mfs.core.finder import get_finder

from .tables import ELEMENTS_FOR, ION_TO_ADDUCT


@dataclass
class PeakMatch:
    """
    One matched MS2 subpeak
    """
    mz: float
    intensity: float
    formula: str          # neutral subformula, Hill string (MIST-parseable)
    mass: float           # neutral subformula's monoisotopic mass
    ppm: float


@dataclass
class SpectrumAssignment:
    """
    All subpeak matches for one (root_formula, ion) candidate
    """
    root_formula: str
    ion: str
    matches: list[PeakMatch] = field(default_factory=list)


def _query_kwargs(
        max_counts: str,
        ion: str,
        frag_ppm: float,
        filter_rdbe: tuple[float, float],
        check_octet: bool,
):
    adduct, charge = ION_TO_ADDUCT[ion]
    return dict(
        charge=charge,
        adduct=adduct,
        max_counts=max_counts,  # subset-of-budget: unmentioned elements capped at 0
        error_ppm=frag_ppm,
        filter_rdbe=filter_rdbe,
        check_octet=check_octet,
    )


def assign_spectrum(
    spec: SpectrumArray,
    root_formula: str,
    ion: str,
    *,
    use_halogens: bool,
    frag_ppm: float = 10.0,
    filter_rdbe: tuple[float, float] = (0.0, 100.0),
    check_octet: bool = True,
) -> SpectrumAssignment:
    """
    Assign subformulae to the peaks of `spec` under `root_formula`+`ion`

    `spec` is expected base-peak normalized (see `spectrum.build_spectrum`)
    """
    if ion not in ION_TO_ADDUCT:
        raise ValueError(
            f"Unknown ion {ion!r}; expected one of {sorted(ION_TO_ADDUCT)}"
        )
    finder = get_finder(ELEMENTS_FOR[bool(use_halogens)])
    qkw = _query_kwargs(
        root_formula,
        ion,
        frag_ppm,
        filter_rdbe,
        check_octet,
    )   # subset-of-root: unmentioned elements capped at 0

    return _assign_monoisotopic(
        spec,
        root_formula,
        ion,
        finder,
        qkw,
    )


def _assign_monoisotopic(
        spec: SpectrumArray,
        root_formula: str,
        ion: str,
        finder: FormulaFinder,
        qkw: dict,
) -> SpectrumAssignment:
    """
    Nearest-monoisotopic subformula per peak (old-pipeline semantics)

    # TODO: selects mass error mf for each peak. Not smart. Revisit
    """
    out = SpectrumAssignment(
        root_formula=root_formula,
        ion=ion,
    )

    # dedup by formula, summing intensity
    seen: dict[str, PeakMatch] = {}
    for i in range(spec.shape[0]):
        res = finder.find_formulae(
            mass=float(spec["mz"][i]),
            **qkw,
        )

        if len(res) == 0:
            continue

        best = res[0]  # by default sorted by |error|

        formula = best.formula.formula
        if formula in seen:
            seen[formula].intensity += float(spec["intsy"][i])
            continue
        match = PeakMatch(
            mz=float(spec["mz"][i]),
            intensity=float(spec["intsy"][i]),
            formula=formula,
            mass=best.formula.monoisotopic_mass,
            ppm=float(best.error_ppm),
        )
        seen[formula] = match
        out.matches.append(match)
    return out


# Cross-root batch assignment:

# Enumerate each peak's subformulae ONCE at the element-wise-max budget over
# the candidate roots, then reduce per root to a cheap `counts <= root`
# integer filter. This is much cheaper than one `find_formulae` per (root, peak)
# Output is identical to looping `assign_spectrum` per root (see
# `tests/test_assignment_caching.py`).

@lru_cache(maxsize=None)
def _root_count_vec_cached(
        root_formula: str,
        symbols: tuple,
) -> tuple:
    idx = {str(s): i for i, s in enumerate(symbols)}
    vec = [0] * len(symbols)
    for sym, item in molmass.Formula(root_formula).composition().items():
        col = idx.get(str(sym))
        if col is not None:
            vec[col] = item.count
    return tuple(vec)


def _root_count_vec(
        root_formula: str,
        symbols,
) -> np.ndarray:
    """
    Element-count vector for `root_formula` aligned to `symbols` order

    The same root formula string recurs heavily across spectra/ions (decoy
    pools overlap for similar precursor masses), so the actual molmass parse +
    composition() walk is cached by (formula, symbols) -- a pure, deterministic
    computation. Returns a fresh array each call so callers can't mutate the
    cached entry.
    """
    return np.array(
        _root_count_vec_cached(root_formula, tuple(str(s) for s in symbols)),
        dtype=np.int64,
    )


def assign_spectrum_batch(
    spec: SpectrumArray,
    roots,
    ion: str,
    *,
    use_halogens: bool,
    frag_ppm: float = 10.0,
    filter_rdbe: tuple[float, float] = (0.0, 100.0),
    check_octet: bool = True,
) -> dict[str, SpectrumAssignment]:
    """
    Assign subformulae to `spec` for many candidate roots sharing one `ion`

    Enumerates each peak once at the element-wise-max budget over `roots`,
    then filters per root.

    Returns `{root_formula: SpectrumAssignment}`.

    Equivalent to calling `assign_spectrum` once per root, but far fewer
     mass decomposition queries. `spec` is expected base-peak normalized.
    """
    if ion not in ION_TO_ADDUCT:
        raise ValueError(
            f"Unknown ion {ion!r}; expected one of {sorted(ION_TO_ADDUCT)}"
        )
    roots = list(dict.fromkeys(roots))  # dedup, preserve order
    finder = get_finder(ELEMENTS_FOR[bool(use_halogens)])
    symbols = finder._symbols

    root_vecs = {r: _root_count_vec(r, symbols) for r in roots}
    if root_vecs:
        budget_vec = np.max(np.stack(list(root_vecs.values())), axis=0)
    else:
        budget_vec = np.zeros(len(symbols), dtype=np.int64)
    # Full symbol list with explicit zeros -> guaranteed superset of every root.
    budget = {str(s): int(budget_vec[i]) for i, s in enumerate(symbols)}

    qkw = _query_kwargs(budget, ion, frag_ppm, filter_rdbe, check_octet)

    # Enumerate each peak ONCE at the shared budget (root-independent).
    peak_enum = []  # per peak: (mz, intsy, res, counts|None, error_ppm|None)
    for i in range(spec.shape[0]):
        mz = float(spec["mz"][i])
        intsy = float(spec["intsy"][i])
        res = finder.find_formulae(mass=mz, **qkw)
        b = res._backend
        if len(b):
            peak_enum.append((mz, intsy, res, b._counts, np.asarray(b._error_ppm)))
        else:
            peak_enum.append((mz, intsy, res, None, None))

    return {
        root: _assign_batch_monoisotopic(root, ion, root_vecs[root], peak_enum)
        for root in roots
    }


def _assign_batch_monoisotopic(
        root_formula,
        ion,
        root_vec,
        peak_enum
) -> SpectrumAssignment:
    """
    Per-root monoisotopic reduction over the cached per-peak enumeration
    """
    out = SpectrumAssignment(root_formula=root_formula, ion=ion)
    seen: dict[str, PeakMatch] = {}
    for mz, intsy, res, counts, _ in peak_enum:
        if counts is None:
            continue
        keep = np.flatnonzero((counts <= root_vec).all(axis=1))
        if keep.size == 0:
            continue
        best = res[int(keep[0])]  # backend sorted by |error| -> first survivor is best
        formula = best.formula.formula
        if formula in seen:
            seen[formula].intensity += intsy
            continue
        match = PeakMatch(
            mz=mz, intensity=intsy, formula=formula,
            mass=best.formula.monoisotopic_mass, ppm=float(best.error_ppm),
        )
        seen[formula] = match
        out.matches.append(match)
    return out
