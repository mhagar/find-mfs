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
  with per-peak erfc tail probabilities (a mass-deviation term and an
   intensity-log-ratio term), each with an intensity-dependent sigma. These
    per-term p-values are combined into a single envelope-level p-value with
     Fisher's method, so a good fit scores ~-1 regardless of how many peaks the
      envelope has (a plain sum would drift by ~-1 per scored term, letting large
       envelopes like Br4 swamp every other term in the posterior).

Non-monoisotopic peaks are scored in *mass- difference space* anchored on M0
(M0 -> 0), because the relative spacing is measured far more accurately than
 the absolute mass. The M0's absolute mass error is NOT scored here
  it belongs to the separate `mass_loglik` term, so the calibration offset
   is never penalized twice.
"""
from __future__ import annotations

import math

import numpy as np
from molmass import Formula
from scipy.stats import chi2

from ..isotopes.envelope import get_isotope_envelope
from ..core.light_formula import LightFormula

# --- floors / thresholds ------------------------------------------------------
_MIN_REL_INT = 0.02          # predicted peaks below 2% rel are ignored when scoring
_SIM_MZ_TOLERANCE = 0.05     # Combine peaks within this mz tolerance
_SIM_INTENSITY_THRESHOLD = 0.001  # simulate down to this rel intensity, regardless
                                   # of `min_rel` -- keeps the predicted envelope a
                                   # pure function of the formula, not of the
                                   # caller's scoring-time leniency
_LOGPROB_FLOOR = -25.0       # per-term and combined log-prob floor (avoids -inf)
_EPS_INT = 1e-6              # intensity floor for a predicted-but-absent peak

# --- intensity-dependent sigmas (piecewise-linear) -------------
# alpha inflates the mass sigma and beta inflates the intensity sigma as peaks
# get weaker. Anchors are (relative_intensity, multiplier),
# interpolated and clamped at the ends.
# TODO: Calibrate these later
_ALPHA_ANCHORS = np.array(
    [[1.00, 1.0],
     [0.20, 1.5],
     [0.02, 3.0]]
)
_BETA_ANCHORS = np.array(
    [[1.00, 0.08],
     [0.20, 0.30],
     [0.02, 1.00]]
)

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
    return (ppm_da / 3.0) * _interp_decreasing(
        _ALPHA_ANCHORS,
        f
    )


def _sigma_int(
        f: float
) -> float:
    """
    Intensity sigma: (1/3) * log(1 + beta(f)).
    """
    return (1.0 / 3.0) * math.log1p(
        _interp_decreasing(_BETA_ANCHORS, f)
    )


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
    ion_formula: Formula | LightFormula | None,
    ms1_peaks: np.ndarray,
    *,
    ppm: float = 5.0,
    mz_match_da: float = 0.02,
    min_rel: float = _MIN_REL_INT,
) -> float:
    """
    Isotope-pattern log-likelihood for a candidate against an
    observed MS1 peak list.

    Args:
        ion_formula: formula to simulate an isotope envelope for
        ms1_peaks: SpectrumArray of observed peaks (a full scan or a
            feature's grouped peaks).

        ppm: mass tolerance taken as the ~3-sigma bound for the mass term.
        mz_match_da: half-width of the m/z search window for matching a
            predicted peak to an observed one. Should be generous.

        min_rel: predicted peaks below this relative intensity are ignored.

    Returns:
        The log of the Fisher-combined p-value over all scored terms (higher /
         closer to 0 is a better match; ~-1 on average for a correct formula,
         0 when nothing beyond the reference peak is scored),
         or _LOGPROB_FLOOR when there is no usable MS1 signal
            (no peaks, empty pred envelope, predicted M0 peak isn't observed)
         or _LOGPROB_FLOOR when ion_formula is None
            (which can happen if adduct > core formula)
    """
    if not ion_formula:
        return _LOGPROB_FLOOR

    predicted_envelope: np.ndarray = get_isotope_envelope(
        formula=ion_formula,
        mz_tolerance=_SIM_MZ_TOLERANCE,
        threshold=_SIM_INTENSITY_THRESHOLD,
    )

    if (
        ms1_peaks is None
        or ms1_peaks.dtype.names is None
        or ms1_peaks.shape[0] == 0
    ):
        raise ValueError(f"Invalid ms1_peaks array: {ms1_peaks}")

    mzs, ints = ms1_peaks['mz'], ms1_peaks['intsy']

    pred = np.asarray(
        predicted_envelope,
        dtype=float,
    )
    if pred.ndim != 2 or pred.shape[0] == 0 or pred.shape[1] < 2:
        return _LOGPROB_FLOOR
    pred = pred[pred[:, 1] >= min_rel]
    if pred.shape[0] < 1:
        return _LOGPROB_FLOOR

    pred_mz, pred_int = pred[:, 0], pred[:, 1]

    # Match each predicted peak to the tallest observed signal in an m/z window.
    obs_mz = np.full(pred.shape[0], np.nan)
    obs_int = np.zeros(pred.shape[0])
    for k in range(pred.shape[0]):
        omz, oint = _tallest_in_window(mzs, ints, pred_mz[k], mz_match_da)
        if omz is not None:
            obs_mz[k], obs_int[k] = omz, oint

    # Reference = the tallest *predicted* peak, not M0. For heavily halogenated
    # ions (e.g. Br4) M0 is weak and its measurement noise would contaminate every
    # peak's ratio and drive the intensity sigma into its tightest clamp; the
    # tallest peak is high-SNR, essentially always observed, and matches the sigma
    # anchors (1.0 = the strong reference). For M0-dominated envelopes this IS M0.
    ref = int(np.argmax(pred_int))
    if np.isnan(obs_mz[ref]) or obs_int[ref] <= 0:
        return _LOGPROB_FLOOR  # reference (tallest) peak not observed -> unusable

    # Normalize both envelopes to the reference peak (scale-invariant ratios,
    # undistorted by missing peaks).
    pred_rel = pred_int / pred_int[ref]
    obs_rel = obs_int / obs_int[ref]

    matched = ~np.isnan(obs_mz)
    # Detection floor: the weakest peak we actually observed (reference-relative). A
    # predicted peak that is absent AND below this floor is simply below the
    # demonstrated detection limit -- not evidence against the formula -- so it is
    # not penalized. This lets a truncated or partially-selected envelope score like
    # the full one, while a predicted peak that *should* have been visible but is
    # missing is still penalized as a real gap.
    obs_floor = float(obs_rel[matched].min())

    obs_mz_ref = obs_mz[ref]
    pred_mz_ref = pred_mz[ref]

    logps: list[float] = []
    for k in range(pred.shape[0]):
        # Skip predicted peaks that are absent and below what we could detect.
        if not matched[k] and pred_rel[k] < obs_floor:
            continue

        f = obs_rel[k]  # observed intensity relative to the reference (0 if absent)
        p = pred_rel[k]

        if k == ref:
            # The reference peak is 1:1 by construction
            # and its absolute mass error is considered in mass_loglik
            # counting it would only inflate Fisher's degrees of freedom.
            continue

        # Intensity term.
        # An absent-but-expected peak has
        # f ~ 0 -> large log-ratio -> a floored penalty, which is correct.
        f_eff = f if f > 0 else _EPS_INT
        logps.append(_log_erfc_prob(
            math.log(f_eff / p),
            _sigma_int(min(f, 1.0)),
        ))

        # Mass term for matched peaks.
        # Scored in reference-anchored relative m/z (more accurate)
        if not matched[k]:
            continue
        obs_diff = obs_mz[k] - obs_mz_ref
        pred_diff = pred_mz[k] - pred_mz_ref
        logps.append(_log_erfc_prob(
            obs_diff - pred_diff, _sigma_mass_da(f, pred_mz[k], ppm)
        ))

    if not logps:
        return 0.0  # only the reference peak was scored -> no isotope evidence

    # Fisher's method: under a correct formula each term's p-value is ~U(0,1), so
    # -2 * sum(log p) ~ chi2(2k). Its survival function is the envelope-level p-value.
    stat = -2.0 * sum(logps)
    return max(float(chi2.logsf(stat, 2 * len(logps))), _LOGPROB_FLOOR)
