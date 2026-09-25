"""
Tests for halogen detection
"""
import numpy as np
import pytest
from molmass import Formula
from find_mfs.spectra.halogen import envelope_is_halogen
from find_mfs.isotopes.envelope import get_isotope_envelope
from find_mfs.spectra.envelopes import to_spec_arr, ISOTOPE_SPACING

THIAMPHENICOL = "C12H15Cl2NO5S"

class TestHalogenDetection:
    def test_envelope_is_halogen(self):
        thiamphenicol = Formula(THIAMPHENICOL)
        envelope = get_isotope_envelope(
            thiamphenicol,
            mz_tolerance=0.05,
            threshold=0.001,
        )

        spec_arr = to_spec_arr(
            envelope[:,0],
            envelope[:,1],
        )

        assert envelope_is_halogen(spec_arr), \
            "Thiamphenicol envelope should be identified as chlorinated"

    def test_bromine_detected_without_m1(self):
        """Br has a strong M+2 but a weak M+1 (13C) that can fall below a noise
        floor. Detection must not require M+1 to be present."""
        env = get_isotope_envelope(
            Formula("C10H10Br2"), mz_tolerance=0.02, threshold=0.001
        )
        env = env[np.argsort(env[:, 0])]
        m0 = env[0, 0]
        # Drop the weak M+1 (13C), keep M0 and the strong Br M+2.
        keep = np.abs(env[:, 0] - (m0 + ISOTOPE_SPACING)) > 0.1
        env = env[keep]
        spec_arr = to_spec_arr(env[:, 0], env[:, 1])
        assert envelope_is_halogen(spec_arr), \
            "Br2 envelope should be detected even with M+1 absent"



# --- robustness to realistic measurement error --------------------------------
#
# Simulated [M+H]+ envelopes with per-peak m/z error (5 ppm) and intensity noise
# (10% relative), seeded. The old detector searched for M+2 at the 13C2 offset
# (2.0067 Da) with a hard 2.0 Da cut-off and required M2 > M1, which missed most
# halogenated ions at 5 ppm and every large monochlorinated one outright.

N_TRIALS = 100

HALOGENATED = [
    "C9H11ClNO2",        # 1 Cl, small
    "C20H23ClN3O4",      # 1 Cl, mid
    "C40H51ClN3O8",      # 1 Cl, large: M+1 ~ M+2, so no zig-zag
    "C12H16Cl2NO5S",     # thiamphenicol [M+H]+
    "C20H23BrN3O4",      # 1 Br
    "C45H61BrN3O8",      # 1 Br, large
    "C10H7Br4O2",        # Br4
]

NOT_HALOGENATED = [
    "C20H25N2O4S2",      # S2: 34S adds real M+2
    "C18H21N3O3S3",      # S3
    "C60H81N3O20",       # O-rich
    "C80H111N5O10",      # carbon-rich: big M+1 and M+2
]


def _nominal_envelope(formula: str) -> tuple[np.ndarray, np.ndarray]:
    """Unit-resolution envelope (fine structure merged), [M+H]+ convention."""
    env = get_isotope_envelope(Formula(formula), mz_tolerance=0.05, threshold=0.002)
    env = env[np.argsort(env[:, 0])]
    return env[:, 0] - 0.000549, env[:, 1] / env[:, 1].max()


def _detection_rate(formula: str, ppm: float, intsy_noise: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    mz, intsy = _nominal_envelope(formula)
    hits = 0
    for _ in range(N_TRIALS):
        noisy_mz = mz * (1 + rng.normal(0, ppm, mz.size) * 1e-6)
        noisy_intsy = intsy * (1 + rng.normal(0, intsy_noise, intsy.size))
        hits += envelope_is_halogen(to_spec_arr(noisy_mz, noisy_intsy), ppm=ppm)
    return hits / N_TRIALS


@pytest.mark.parametrize("formula", HALOGENATED)
def test_halogen_detected_under_noise(formula):
    rate = _detection_rate(formula, ppm=5.0, intsy_noise=0.10, seed=0)
    assert rate >= 0.95, f"{formula}: detected in only {rate:.0%} of trials"


@pytest.mark.parametrize("formula", NOT_HALOGENATED)
def test_no_false_halogen_under_noise(formula):
    rate = _detection_rate(formula, ppm=5.0, intsy_noise=0.10, seed=0)
    assert rate <= 0.05, f"{formula}: falsely flagged in {rate:.0%} of trials"


def test_large_monochloride_without_zigzag():
    """Noiseless C40 monochloride: M+1 ~ M+2, so the old 'M2 > M1' rule failed."""
    mz, intsy = _nominal_envelope("C40H51ClN3O8")
    assert intsy[1] >= intsy[2] * 0.95          # no zig-zag to speak of
    assert envelope_is_halogen(to_spec_arr(mz, intsy))


def test_m2_found_at_chlorine_offset_not_13c2():
    """M+2 measured 5 mDa low (1.992 Da from M0) is still a Cl M+2. The old
    detector's window only spanned [1.9967, 2.000] Da."""
    mz = np.array([500.0, 501.0034, 501.992])
    intsy = np.array([1.0, 0.25, 0.36])
    assert envelope_is_halogen(to_spec_arr(mz, intsy))


def test_resolved_fine_structure_is_summed():
    """At high resolution M+2 splits into 37Cl and 13C2/18O peaks. Summing them
    must not break detection, and 13C2 alone must not trigger it."""
    env = get_isotope_envelope(Formula("C20H23ClN3O4"), mz_tolerance=0.001, threshold=0.001)
    assert envelope_is_halogen(to_spec_arr(env[:, 0], env[:, 1]))

    env = get_isotope_envelope(Formula("C40H60N3O8"), mz_tolerance=0.001, threshold=0.001)
    assert not envelope_is_halogen(to_spec_arr(env[:, 0], env[:, 1]))


def test_doubly_charged_envelope():
    """Offsets scale with 1/z."""
    mz, intsy = _nominal_envelope("C20H23ClN3O4")
    mz_2plus = (mz + 1.007276) / 2           # [M+2H]2+ from the [M+H]+ positions
    spec = to_spec_arr(mz_2plus, intsy)
    assert envelope_is_halogen(spec, charge=2)
    assert not envelope_is_halogen(spec, charge=1)


def test_ppm_widens_tolerance():
    """At m/z 1500, a 15 mDa M+2 miss is outside the 0.01 Da floor but inside
    3 sigma of a 5 ppm difference error (~32 mDa)."""
    mz = np.array([1500.0, 1501.0034, 1500.0 + 1.9975 - 0.015])
    intsy = np.array([1.0, 0.9, 0.9])
    spec = to_spec_arr(mz, intsy)
    assert not envelope_is_halogen(spec, tol=0.01)
    assert envelope_is_halogen(spec, tol=0.01, ppm=5.0)
