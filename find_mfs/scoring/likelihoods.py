"""
MS1 likelihood terms for scoring a candidate formula against an
observed precursor isotope envelope.

These are the *data* terms in the stacked log-posterior (as opposed to the
chemical-composition prior in ``prior.py``):

    log_posterior = chem_logprior                # P(formula), from the GMM prior
                  + iso_weight  * iso_loglik      # P(envelope | formula)
                  + mass_weight * mass_loglik     # P(precursor | formula)

`isotope_loglik`: for each predicted isotopologue peak we search a small m/z
 window in the raw MS1 scan, take the tallest signal there, and score the match
  as a product of per-peak erfc tail likelihoods (a mass-deviation term and an
   intensity-log-ratio term), each with an intensity-dependent sigma. The total
    is the sum of per-peak log-probs.

Non-monoisotopic peaks are scored in *mass- difference space* anchored on M0
(M0 -> 0), because the relative spacing is measured far more accurately than
 the absolute mass. The M0's absolute mass error is NOT scored here
  it belongs to the separate `mass_loglik` term, so the calibration offset
   is never penalized twice.
"""
from __future__ import annotations

import math

import numpy as np

# --- floors / thresholds ------------------------------------------------------
_MIN_REL_INT = 0.02      # predicted peaks below 2% rel are ignored
_LOGPROB_FLOOR = -25.0   # per-peak log-prob floor (avoids -inf on erfc->0)
_EPS_INT = 1e-6          # intensity floor for a predicted-but-absent peak

# --- intensity-dependent sigmas (piecewise-linear) -------------
# alpha inflates the mass sigma and beta inflates the intensity sigma as peaks
# get weaker. Anchors are (relative_intensity, multiplier),
# interpolated and clamped at the ends.
# TODO: Calibrate these later
_ALPHA_ANCHORS = np.array([[1.00, 1.0], [0.20, 1.5], [0.02, 3.0]])
_BETA_ANCHORS = np.array([[1.00, 0.08], [0.20, 0.30], [0.02, 1.00]])


def _interp_decreasing(
        anchors: np.ndarray,
        f: float,
) -> float:
    """
    Piecewise-linear interpolation of `anchors` (given high->low intensity),
    clamped outside the anchor range.
    """
    xs = anchors[::-1, 0]  # ascending intensity
    ys = anchors[::-1, 1]
    return float(np.interp(f, xs, ys))


def _sigma_mass_da(
        f: float,
        mz: float,
        ppm: float,
) -> float:
    """
    Mass sigma in Da.

    `ppm` is taken as the ~3-sigma bound, so
    sigma = (ppm_in_da / 3) * alpha(f)
    """
    ppm_da = ppm * 1e-6 * mz
    return (ppm_da / 3.0) * _interp_decreasing(_ALPHA_ANCHORS, f)


def _sigma_int(
        f: float
) -> float:
    """
    Intensity sigma: (1/3) * log(1 + beta(f)).
    """
    return (1.0 / 3.0) * math.log1p(_interp_decreasing(_BETA_ANCHORS, f))


def _log_erfc_prob(
        deviation: float,
        sigma: float
) -> float:
    """
    log of the two-sided tail likelihood
    erfc(|deviation| / (sqrt(2) sigma)), floored to avoid -inf.
    """
    if sigma <= 0:
        return 0.0 if deviation == 0 else _LOGPROB_FLOOR
    p = math.erfc(abs(deviation) / (math.sqrt(2.0) * sigma))
    if p <= 0.0:
        return _LOGPROB_FLOOR
    return max(math.log(p), _LOGPROB_FLOOR)


def _tallest_in_window(
    mzs: np.ndarray,
    ints: np.ndarray,
    target: float,
    tol_da: float,
) -> tuple[float | None, float]:
    """
    (m/z, intensity) of the tallest observed peak within `tol_da` of
    `target`, or `(None, 0.0)` if none.

    i.e. search a region, take the tallest signal
    """
    lo, hi = target - tol_da, target + tol_da
    idx = np.flatnonzero((mzs >= lo) & (mzs <= hi))
    if idx.size == 0:
        return None, 0.0
    j = idx[int(np.argmax(ints[idx]))]
    return float(mzs[j]), float(ints[j])


def mass_loglik(
        error_ppm: float,
        sigma_ppm: float = 5.0,
) -> float:
    """
    Gaussian log-likelihood of the precursor mass error (up to a constant):
    `-0.5 * (error_ppm / sigma_ppm)^2`

    Uses the candidate's already-computed ppm error, so no theoretical-mass
    recomputation is needed.
    """
    return -0.5 * (error_ppm / sigma_ppm) ** 2


def isotope_loglik(
    predicted_envelope: np.ndarray,
    ms1_peaks: np.ndarray,
    *,
    ppm: float = 5.0,
    mz_match_da: float = 0.02,
    min_rel: float = _MIN_REL_INT,
) -> float | None:
    """
    Isotope-pattern log-likelihood for a candidate against the raw
    MS1 peak list.

    Args:
        predicted_envelope: `(n, 2)` array of `[m/z, rel_intensity]` for the
            candidate *ion* (as produced by `get_isotope_envelope`).

            Peaks below `min_rel` are pruned here.

        ms1_peaks: `(m, 2)` array of raw observed `[m/z, intensity]` peaks
            (a full scan or a feature's grouped peaks).

        ppm: mass tolerance taken as the ~3-sigma bound for the mass term.
        mz_match_da: half-width of the m/z search window for matching a
            predicted peak to an observed one.

        min_rel: predicted peaks below this relative intensity are ignored.

    Returns:
        The isotope log-likelihood (higher / closer to 0 is a better match),
         or `None` when there is no usable MS1 signal - no peaks, an empty
        predicted envelope, or the predicted M0 peak isn't present in the scan
        (in which case the MS1 can't speak to this candidate).
    """
    if ms1_peaks is None or predicted_envelope is None:
        return None

    peaks = np.asarray(ms1_peaks, dtype=float)
    if peaks.ndim != 2 or peaks.shape[0] == 0 or peaks.shape[1] < 2:
        return None
    mzs, ints = peaks[:, 0], peaks[:, 1]

    pred = np.asarray(predicted_envelope, dtype=float)
    if pred.ndim != 2 or pred.shape[0] == 0 or pred.shape[1] < 2:
        return None
    pred = pred[pred[:, 1] >= min_rel]
    if pred.shape[0] < 1:
        return None

    pred_mz, pred_int = pred[:, 0], pred[:, 1]
    m0_i = int(np.argmin(pred_mz))  # monoisotopic = lowest-m/z predicted peak

    # Match each predicted peak to the tallest observed signal in an m/z window.
    obs_mz = np.full(pred.shape[0], np.nan)
    obs_int = np.zeros(pred.shape[0])
    for k in range(pred.shape[0]):
        omz, oint = _tallest_in_window(mzs, ints, pred_mz[k], mz_match_da)
        if omz is not None:
            obs_mz[k], obs_int[k] = omz, oint

    if np.isnan(obs_mz[m0_i]):
        return None  # precursor M0 not in this MS1 scan -> unusable

    # Sum-normalize both envelopes
    pred_norm = pred_int / pred_int.sum()
    obs_sum = obs_int.sum()
    if obs_sum <= 0:
        return None
    obs_norm = obs_int / obs_sum

    obs_mz0 = obs_mz[m0_i]
    pred_mz0 = pred_mz[m0_i]

    total = 0.0
    for k in range(pred.shape[0]):
        f = obs_norm[k]  # observed relative intensity of this peak

        # Intensity term (all peaks, incl. M0). A predicted-but-absent peak has
        # f ~ 0 -> large log-ratio -> a floored penalty, which is correct.
        p = pred_norm[k]
        f_eff = f if f > 0 else _EPS_INT
        total += _log_erfc_prob(math.log(f_eff / p), _sigma_int(f))

        # Mass term: skip M0 (its absolute error is the mass_loglik's job) and
        # any absent peak. Non-M0 peaks are scored in M0-anchored difference
        # space, where relative spacing is measured far more accurately.
        if k == m0_i or np.isnan(obs_mz[k]):
            continue
        obs_diff = obs_mz[k] - obs_mz0
        pred_diff = pred_mz[k] - pred_mz0
        total += _log_erfc_prob(
            obs_diff - pred_diff, _sigma_mass_da(f, pred_mz[k], ppm)
        )

    return total
