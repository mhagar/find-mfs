"""
Raw MS2 reranker logits for a set of candidate formulae.

Chains the pieces in this package: assign subformulae to the MS2 peaks under
each candidate root, featurize, and run the reranker.

## Why this returns logits and not log P(formula | MS2)

MistNet is trained with a softmax-over-candidates NLL, so `softmax(logits)`
genuinely is `P(formula | MS2)` -- but only relative to the candidate set the
softmax ran over. That makes the normalized value set-dependent: it cannot be
cached per formula, and is only valid for a given *set* of mf candidates

So the split is:

* **here**: the per-candidate logit, which is meaningful on its own and stays
  valid through sorting and filtering. Stored as `FormulaCandidate.ms2_logit`.
* **`FormulaSearchResults.ms2_loglik()`**: the normalization, recomputed over
  whatever candidates are currently present

## Cost

~O(n_candidates) network evaluations with an O(peaks^2) constant, which is
far more expensive than the chem / mass / isotope terms. Callers should gate it
to the top candidates under the cheap terms
(see `ms2_top_n` in `find_mfs.scoring.FormulaScorer.score`).
"""

from __future__ import annotations

import numpy as np

from find_mfs.spectra import spec_from_pairs

from .assign import assign_spectrum_batch
from .featurize import Featurizer, score_candidates
from .tables import ION_TO_ADDUCT

# (find-mfs adduct, charge) -> MIST-CF ion string. The reranker's ion one-hot is
# a fixed 7-ion positive-mode vocabulary, so candidates outside it (notably all
# of negative mode) simply get no MS2 term.
ADDUCT_TO_ION = {v: k for k, v in ION_TO_ADDUCT.items()}


def resolve_ion(
        adduct: str | None,
        charge: int = 1,
) -> str | None:
    """
    Map a find-mfs `(adduct, charge)` to a reranker ion string, or None.

    None means the reranker has no column for this ion and the candidate simply
    gets no MS2 term -- it is not an error.
    """
    return ADDUCT_TO_ION.get((adduct, charge))


def ms2_logits(
    model,
    formulae: list[str],
    ion: str,
    ms2_peaks: np.ndarray,
    *,
    precursor_mz: float,
    instrument: str = "unknown",
    frag_ppm: float = 10.0,
    use_halogens: bool = True,
    check_octet: bool = True,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Raw reranker logit for each of `formulae`, as a `(n,)` array.

    These are *unnormalized*. To turn them into log P(formula | MS2), assign
    them to candidates as `ms2_logit` and use
    `FormulaSearchResults.ms2_loglik()`, which normalizes over the current
    candidate set (see the module docstring for why that split exists).

    Args:
        model: a `MistNetNumpy`.
        formulae: candidate *neutral core* formulae (Hill strings). Duplicates
            are allowed; each is scored once and the result broadcast back.
        ion: MIST-CF ion string, e.g. `"[M+H]+"` (see `resolve_ion`).
        ms2_peaks: `(n, 2)` array of `[m/z, intensity]`. Expected already
            de-isotoped and precursor-cropped; it is base-peak normalized here.
        precursor_mz: observed precursor m/z. Only affects the result when the
            checkpoint was trained with `cls_mass_diff`.
        instrument: instrument name for the one-hot (see `INSTRUMENT_TO_TYPE`).
        frag_ppm: fragment mass tolerance for subformula assignment.
        use_halogens: search CHNOPS+FClBrI rather than CHNOPS.
        check_octet: apply the octet rule to candidate subformulae.
        batch_size: candidates per forward pass.

    Returns:
        Array aligned to `formulae`. Empty if `formulae` is empty.
    """
    if not formulae:
        return np.zeros(0)

    spec = spec_from_pairs(ms2_peaks)
    # dict.fromkeys dedups while preserving order; the assigner needs unique
    # roots but callers may legitimately pass repeats.
    unique = list(dict.fromkeys(formulae))
    assignments = assign_spectrum_batch(
        spec, unique, ion,
        use_halogens=use_halogens, frag_ppm=frag_ppm, check_octet=check_octet,
    )

    feat = Featurizer.from_model(model)
    cands = [
        feat.candidate(
            assignments[f], parentmass=precursor_mz, instrument=instrument, name=f
        )
        for f in unique
    ]
    logits = score_candidates(model, cands, batch_size=batch_size)

    by_formula = dict(zip(unique, logits))
    return np.array([by_formula[f] for f in formulae])
