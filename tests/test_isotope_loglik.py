"""
Tests for the isotope-pattern log-likelihood, focused on robustness to an
*incomplete* observed envelope (a partial selection or truncated discovery must
not catastrophically penalise the correct formula) while retaining discrimination
against formulae whose predicted pattern genuinely contradicts the observation.
"""
import numpy as np
from molmass import Formula

from find_mfs.scoring.likelihoods import isotope_loglik
from find_mfs.isotopes.envelope import get_isotope_envelope
from find_mfs.spectra.envelopes import to_spec_arr

ION = Formula("[C12H16Cl2NO5S]+")   # thiamphenicol [M+H]+, a strong Cl2 pattern


def _predicted():
    env = get_isotope_envelope(ION, mz_tolerance=0.02, threshold=0.001)
    return env[np.argsort(env[:, 0])]


def _select_tallest(env, n):
    idx = np.sort(np.argsort(env[:, 1])[::-1][:n])
    return to_spec_arr(env[idx, 0].tolist(), env[idx, 1].tolist())


def test_full_envelope_scores_near_zero():
    env = _predicted()
    full = to_spec_arr(env[:, 0].tolist(), env[:, 1].tolist())
    assert isotope_loglik(ION, full, ppm=8.0) > -1.0


def test_partial_selection_not_penalized():
    """Selecting the N tallest peaks (how a user selects) must score like the full
    envelope -- this is the reported bug."""
    env = _predicted()
    for n in (2, 3, 4):
        sub = _select_tallest(env, n)
        assert isotope_loglik(ION, sub, ppm=8.0) > -1.0, f"{n} peaks over-penalized"


def test_single_peak_is_neutral_not_catastrophic():
    """M0 alone carries no isotope information -> ~0, not a huge negative."""
    env = _predicted()
    m0 = to_spec_arr([env[0, 0]], [env[0, 1]])
    assert isotope_loglik(ION, m0, ppm=8.0) > -1.0


def test_discriminates_against_contradictory_pattern():
    """A no-Cl formula predicts a tiny M+2, contradicting an observed strong M+2,
    so it must score clearly worse than the true Cl2 formula on the same peaks."""
    env = _predicted()
    obs3 = to_spec_arr(env[:3, 0].tolist(), env[:3, 1].tolist())  # M0, M+1, strong M+2
    true_ll = isotope_loglik(ION, obs3, ppm=8.0)
    nocl_ll = isotope_loglik(Formula("[C20H28NO5]+"), obs3, ppm=8.0)
    assert true_ll > nocl_ll + 10.0


def test_weak_m0_heavy_halogen_not_catastrophic():
    """Br4: the monoisotopic M0 is weak and the envelope peaks at M+4. Normalizing
    on the tallest peak (not M0) must keep a perfect match near 0, not -20+."""
    br4 = Formula("[C21H24N3O6Br4]+")
    env = get_isotope_envelope(br4, mz_tolerance=0.02, threshold=0.001)
    env = env[np.argsort(env[:, 0])]
    assert env[0, 1] < 0.3   # sanity: M0 really is weak relative to the tallest
    obs = to_spec_arr(env[:, 0].tolist(), env[:, 1].tolist())  # observed == predicted
    assert isotope_loglik(br4, obs, ppm=8.0) > -1.0


def test_absent_expected_peak_is_penalized():
    """Keeping a weak peak while a much stronger predicted peak is absent within the
    detected range IS a real gap and should be penalized."""
    env = _predicted()
    # M0 + weak M+1 only; the strong M+2 (~0.70) is absent though clearly detectable.
    obs = to_spec_arr([env[0, 0], env[1, 0]], [env[0, 1], env[1, 1]])
    assert isotope_loglik(ION, obs, ppm=8.0) < -10.0
