"""
`log P(formula | MS2)` for a set of candidate formulae.

The MS2 term of the stacked log-posterior in :mod:`find_mfs.scoring`.

Chains the pieces in this package: assign subformulae to the MS2 peaks under
 each candidate root, featurize, run the reranker, and normalize across the candidate set.

Two properties distinguish this from the chem / mass / isotope terms:

* **It is set-normalized.** MistNet is trained with a softmax-over-candidates
  NLL, so `softmax(logits)` genuinely is `P(formula | MS2)`

  But that means the value depends on which candidates are in the set,
  cannot be cached per formula, and shifts when candidates are added or removed.

  This is deliberate; do not "fix" it to raw logits.

* **It is expensive**, ~O(n_candidates) network evaluations with an O(peaks^2)
  constant. Callers should gate it to the top candidates under the cheap terms
  (see `ms2_top_n` in :meth:`find_mfs.scoring.FormulaScorer.score`).
"""

from __future__ import annotations

import numpy as np

from .assign import assign_spectrum_batch
from .featurize import Featurizer, score_candidates
from find_mfs.spectra import spec_from_pairs
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
    """
    return ADDUCT_TO_ION.get((adduct, charge))


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    shifted = x - m
    return shifted - np.log(np.exp(shifted).sum())


def ms2_loglik(
    model,
    formulae: list[str],
    ion: str,
    ms2_peaks: np.ndarray,
    *,
    precursor_mz: float,
    instrument: str = "unknown",
    temperature: float = 1.0,
    frag_ppm: float = 10.0,
    use_halogens: bool = True,
    check_octet: bool = True,
    batch_size: int = 256,
) -> np.ndarray:
    """
    `log P(formula | MS2)` for each of `formulae`, as a `(n,)` array.

    Args:
        model: a :class:`~find_mfs.ms2.net.MistNetNumpy`.
        formulae: candidate *neutral core* formulae (Hill strings).
        ion: MIST-CF ion string, e.g. ``"[M+H]+"`` (see :func:`resolve_ion`).
        ms2_peaks: ``(n, 2)`` array of ``[m/z, intensity]``. Expected already
            de-isotoped and precursor-cropped.
        precursor_mz: observed precursor m/z; only used when the checkpoint was
            trained with ``cls_mass_diff``.
        instrument: instrument name for the one-hot (see
            :data:`~find_mfs.ms2.tables.INSTRUMENT_TO_TYPE`).
        temperature: softmax temperature. Lower sharpens the MS2 expert relative
            to the other terms; ~0.5 has been the useful operating point.
        frag_ppm: fragment mass tolerance for subformula assignment.
        use_halogens: search CHNOPS+FClBrI rather than CHNOPS.

    Returns:
        Log-probabilities summing to 1 in probability space. All-zeros if
        `formulae` is empty.
    """
    if not formulae:
        return np.zeros(0)
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    spec = spec_from_pairs(ms2_peaks, normalize=True)
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

    by_formula = dict(zip(unique, log_softmax(logits / temperature)))
    return np.array([by_formula[f] for f in formulae])
