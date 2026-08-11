"""
Spectrum data types, cleaning helpers, and isotope envelope detection
"""
from __future__ import annotations

from typing import Iterable, NewType

import numpy as np
from numpy.typing import NDArray


# -- SpectrumArray type --------------------------------------------------------

SpectrumArray = NewType(
    name='SpectrumArray',
    tp=NDArray[
        np.dtype(  # type: ignore
            [
                ('mz', 'f8'),
                ('intsy', 'f8'),
            ]
        )
    ]
)


def to_spec_arr(
    mz_arr: Iterable[float],
    intsy_arr: Iterable[float],
) -> SpectrumArray:
    result = np.zeros(
        len(mz_arr),
        dtype=[
            ('mz', 'f8'),
            ('intsy', 'f8'),
        ]
    )
    result['mz'] = mz_arr
    result['intsy'] = intsy_arr
    return SpectrumArray(result)


# -- Spectrum cleaning ---------------------------------------------------------
def normalize(
    spec_arr: SpectrumArray,
    base: float = 1.0,
) -> SpectrumArray:
    """
    Scale intensities so the tallest (base) peak equals `base`.

    Base-peak-1.0 is the library-wide convention: it is what the isotope
    matching and the MS2 reranker both expect, and what `build_spectrum`
    produces.

    Args:
        spec_arr: spectrum to normalize. Not modified in place.
        base: intensity assigned to the tallest peak.

    Returns:
        A normalized copy. Returned unchanged if empty or if all
        intensities are non-positive.
    """
    if spec_arr.size == 0:
        return spec_arr

    max_intsy = spec_arr['intsy'].max()
    if max_intsy <= 0:
        return spec_arr

    out = spec_arr.copy()
    out['intsy'] = out['intsy'] / max_intsy * base
    return out


def build_spectrum(
    mz: Iterable[float],
    intsy: Iterable[float],
    *,
    normalize_intensity: bool = True,
    drop_zero: bool = True,
) -> SpectrumArray:
    """
    Build a sorted, base-peak-normalized SpectrumArray from parallel arrays.

    This is the canonical way to get raw peak data into the library. Prefer it
    over calling `to_spec_arr` directly, which does no cleaning and no sorting.

    Args:
        mz: m/z values.
        intsy: intensities, same length as `mz`.
        normalize_intensity: scale so the base peak is 1.0. Leave this on for
            anything that consumes relative intensities -- isotope matching and
            the MS2 reranker both assume the base-peak-1.0 convention, and the
            reranker in particular feeds intensity straight into a trained
            network, so a different scale silently degrades its scores.
        drop_zero: discard non-positive-intensity peaks first.

    Returns:
        SpectrumArray sorted by ascending m/z.

    Raises:
        ValueError: if `mz` and `intsy` have different lengths.
    """
    mz = np.asarray(mz, dtype=np.float64)
    intsy = np.asarray(intsy, dtype=np.float64)
    if mz.shape != intsy.shape:
        raise ValueError(f"mz and intsy length mismatch: {mz.shape} vs {intsy.shape}")

    if drop_zero:
        keep = intsy > 0
        mz, intsy = mz[keep], intsy[keep]

    order = np.argsort(mz, kind="mergesort")
    out = to_spec_arr(mz[order], intsy[order])
    return normalize(out) if normalize_intensity else out


def spec_from_pairs(
    peaks,
    **kwargs,
) -> SpectrumArray:
    """
    Build a SpectrumArray from an `(N, 2)` `[[mz, intsy], ...]` array.

    Thin wrapper around `build_spectrum`; takes the same keyword arguments.
    """
    peaks = np.asarray(peaks, dtype=np.float64)
    if peaks.ndim != 2 or peaks.shape[1] != 2:
        raise ValueError(f"expected (N, 2) [mz, intsy] array, got {peaks.shape}")
    return build_spectrum(peaks[:, 0], peaks[:, 1], **kwargs)


def trim_envelope_left(peaks: SpectrumArray) -> SpectrumArray:
    """
    Trim peaks below the base (tallest) peak in an envelope.

    Discards any peaks with m/z lower than the most intense peak,
    which removes minor in-source fragment signals (e.g. -H2 losses)
    that get incorrectly grouped into the envelope.

    TODO: This assumes monoisotopic = base peak, which fails for
    high-MW compounds (>~1000 Da) where M+1 or M+2 can be tallest.
    """
    if len(peaks) == 0:
        return peaks
    order = np.argsort(peaks['mz'])
    sorted_peaks = peaks[order]
    base_idx = np.argmax(sorted_peaks['intsy'])
    return sorted_peaks[base_idx:]


def make_intensity_mask(
    spec_arr: SpectrumArray,
    min_intensity: float,
) -> NDArray[np.bool_]:
    """
    True for peaks at or above min_intensity
    """
    return spec_arr['intsy'] >= min_intensity


def make_shoulder_mask(
    spec_arr: SpectrumArray,
    tol: float,
) -> NDArray[np.bool_]:
    """
    True for peaks that are not a shoulder of a taller neighbour.
    A peak is a shoulder if another peak within tol Da has higher intensity.
    """
    mz = spec_arr['mz']
    intsy = spec_arr['intsy']
    close = np.abs(mz[:, None] - mz[None, :]) <= tol  # (n, n)
    np.fill_diagonal(close, False)
    has_taller_neighbour = np.any(close & (intsy[None, :] > intsy[:, None]), axis=1)
    return ~has_taller_neighbour


# -- Isotope envelope detection ------------------------------------------------

ISOTOPE_SPACING = 1.0033548  # Da (neutron mass)


def find_peaks_by_spacing(
    spec_arr: SpectrumArray,
    tgt_idx: int,
    tgt_spacing: float,
    max_n: int,
    tol: float,
) -> NDArray[np.intp]:
    """
    Returns indices of peaks at ±n * tgt_spacing from tgt_idx (n = 1..max_n),
    within tolerance tol. Does not include tgt_idx itself.
    """
    tgt_mz: float = spec_arr[tgt_idx]['mz']
    diffs = spec_arr['mz'] - tgt_mz

    match = np.zeros(len(spec_arr), dtype=bool)
    for n in range(1, max_n + 1):
        n_spacing = tgt_spacing * n
        match |= (
            (np.abs(diffs - n_spacing) <= tol)
            | (np.abs(diffs + n_spacing) <= tol)
        )

    return np.where(match)[0]


def find_isotope_envelopes(
    spec_arr: SpectrumArray,
    spacing: float = ISOTOPE_SPACING,
    tol: float = 0.05,
    max_isotopes: int = 8,
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.intp]:
    """
    Greedy isotope envelope detection. Processes peaks tallest-first.

    Returns envelope_labels, shape (len(spec_arr),).
    envelope_labels[i] = envelope ID for peak i, or -1 if unassigned/masked.

    max_isotopes defaults to 8 to accommodate halogen isotopologue patterns.
    mask: bool array, same length as spec_arr. False = excluded from detection.
    """
    n = len(spec_arr)
    envelope_labels = np.full(n, -1, dtype=np.intp)
    if mask is None:
        mask = np.ones(n, dtype=bool)

    intensity_order = np.argsort(spec_arr['intsy'])[::-1]

    envelope_id = 0
    for seed_idx in intensity_order:
        if not mask[seed_idx] or envelope_labels[seed_idx] != -1:
            continue

        neighbors = find_peaks_by_spacing(spec_arr, seed_idx, spacing, max_isotopes, tol)
        unassigned = neighbors[mask[neighbors] & (envelope_labels[neighbors] == -1)]

        envelope_labels[seed_idx] = envelope_id
        envelope_labels[unassigned] = envelope_id
        envelope_id += 1

    return envelope_labels


# -- Envelope statistics -------------------------------------------------------

def _envelope_stats(
    spec_arr: SpectrumArray,
    envelope_labels: NDArray[np.intp],
    n_envelopes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (mono_mz, max_intsy) arrays, one entry per envelope."""
    mono_mz = np.zeros(n_envelopes)
    max_intsy = np.zeros(n_envelopes)
    for eid in range(n_envelopes):
        peaks = spec_arr[envelope_labels == eid]
        mono_mz[eid] = peaks['mz'].min()
        max_intsy[eid] = peaks['intsy'].max()
    return mono_mz, max_intsy
