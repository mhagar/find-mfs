"""
Tests for charge-aware signal grouping (struct-of-arrays).

Covers charge inference (1+ vs 2+ from isotope spacing), singletons, the three
noise-threshold modes, and halogen flagging on real Cl2 data.
"""
from pathlib import Path

import numpy as np
import pytest
from molmass import Formula

from find_mfs.isotopes.envelope import get_isotope_envelope
from find_mfs.spectra import read_mgf
from find_mfs.spectra.envelopes import build_spectrum, ISOTOPE_SPACING
from find_mfs.spectra.grouping import group_signals, NoiseThreshold, H2_LOSS

DATA = Path(__file__).parent / "data" / "thiamphenicol.mgf"


def test_infers_charge_one_and_two_and_singleton():
    z2 = ISOTOPE_SPACING / 2
    mz = [
        300.0, 300.0 + ISOTOPE_SPACING, 300.0 + 2 * ISOTOPE_SPACING,   # 1+ envelope
        500.0, 500.0 + z2, 500.0 + 2 * z2, 500.0 + 3 * z2,             # 2+ envelope (0.5 ladder)
        777.7,                                                          # singleton
    ]
    intsy = [1.0, 0.3, 0.05, 0.8, 0.4, 0.1, 0.04, 0.2]
    g = group_signals(build_spectrum(mz, intsy), max_charge=2)

    assert g.n_groups == 3
    by_mono = {round(g.mono_mz(i), 1): int(g.charge[i]) for i in range(g.n_groups)}
    assert by_mono[300.0] == 1
    assert by_mono[500.0] == 2
    assert by_mono[777.7] == 1

    # Deconvoluted mass of the 2+ group is ~2x its m/z.
    gid_2plus = next(i for i in range(g.n_groups) if round(g.mono_mz(i), 1) == 500.0)
    assert abs(g.deconv_mass[gid_2plus] - 1000.0) < 0.01


def test_stray_half_da_peak_does_not_force_2plus():
    """A single spurious peak 0.5 Da above M0 must NOT flip a 1+ envelope to 2+
    (that artifact surfaces downstream as a bogus [2M+2H]2+ adduct)."""
    z2 = ISOTOPE_SPACING / 2
    mz = [300.0, 300.0 + ISOTOPE_SPACING, 300.0 + z2]   # M0, M+1, and one stray +0.5
    g = group_signals(build_spectrum(mz, [1.0, 0.3, 0.08]))
    base = g.group_containing(300.0)
    assert int(g.charge[base]) == 1


def test_group_labels_index_back_into_spectrum():
    mz = [300.0, 300.0 + ISOTOPE_SPACING, 500.0]
    spec = build_spectrum(mz, [1.0, 0.3, 0.5])
    g = group_signals(spec)
    for gid in range(g.n_groups):
        peaks = g.peaks_of(gid)
        # peaks_of is exactly the spec rows whose label == gid
        assert np.array_equal(peaks, spec[g.group_labels == gid])
        assert g.mono_mz(gid) == spec['mz'][g.mono_idx[gid]]


def test_noise_min_rel_drops_small_peak():
    spec = build_spectrum([100.0, 200.0], [1.0, 0.005])   # base-peak normalized to 1.0
    g = group_signals(spec, noise=NoiseThreshold(min_rel=0.01))
    # The 0.5%-intensity peak is below 1% of base -> unassigned.
    small = np.where(spec['mz'] == 200.0)[0][0]
    assert g.group_labels[small] == -1
    assert g.n_groups == 1


def test_noise_top_n_keeps_only_tallest():
    spec = build_spectrum([100.0, 200.0, 300.0], [1.0, 0.5, 0.2])
    g = group_signals(spec, noise=NoiseThreshold(top_n=1))
    assert g.n_groups == 1
    assert round(g.mono_mz(0), 1) == 100.0


def test_noise_min_abs():
    spec = build_spectrum([100.0, 200.0], [1.0, 0.3])
    g = group_signals(spec, noise=NoiseThreshold(min_abs=0.5))
    assert g.n_groups == 1
    assert round(g.mono_mz(0), 1) == 100.0


def test_halogen_flag_on_thiamphenicol():
    spec = read_mgf(DATA)[0].spec_arr
    g = group_signals(spec, detect_halogens=True)
    base = g.group_containing(356.0126, tol=0.02)   # intact [M+H]+ Cl2 envelope
    assert base is not None
    assert bool(g.is_halogen[base]) is True


# -- In-source -H2 loss must not become the monoisotopic peak ------------------

H2_DATA = Path(__file__).parent / "data" / "h2_loss.mgf"


@pytest.mark.parametrize("noise", [None, NoiseThreshold(min_rel=0.05)])
def test_h2_loss_not_merged_into_envelope(noise):
    """Real spectra: [M-H2+H]+ at seed - 2.0157 was taken as M0, giving the wrong
    precursor m/z and a huge 'M+2' that fired the halogen detector."""
    for spec in read_mgf(H2_DATA):
        precursor = float(spec.metadata['PEPMASS'].split()[0])
        g = group_signals(spec.spec_arr, noise=noise, detect_halogens=True)
        base = g.group_containing(precursor)
        assert g.mono_mz(base) == pytest.approx(precursor)
        assert not g.is_halogen[base]


def _halogen_spectrum(formula: str):
    env = get_isotope_envelope(Formula(formula), mz_tolerance=0.05, threshold=0.002)
    return env[:, 0], env[:, 1] / env[:, 1].max()


@pytest.mark.parametrize("hal", ["Cl", "Br", "ClBr", "Cl2", "Cl3", "Br2", "Cl2Br",
                                 "Br3", "Cl4", "Br4", "Cl4Br3"])
def test_real_halogen_m0_kept(hal):
    """Genuine Cl/Br envelopes with M0 below the tallest peak keep their M0."""
    mz, intsy = _halogen_spectrum(f"C20H30N3O5{hal}")
    g = group_signals(build_spectrum(mz, intsy), detect_halogens=True)
    base = g.group_containing(mz.min())
    assert g.mono_mz(base) == pytest.approx(mz.min())
    assert g.is_halogen[base]


@pytest.mark.parametrize("hal, h2_frac", [("", 0.9), ("Cl", 0.3), ("Br2", 0.3)])
def test_h2_loss_caught_by_mass(hal, h2_frac):
    """Where intensity can't tell (H2 loss ~ parent height, or a halogenated parent
    with a real M+4), the 2.0157 vs 2.0067 spacing decides."""
    mz, intsy = _halogen_spectrum(f"C20H30N3O5{hal}")
    mz = np.concatenate([mz - H2_LOSS, mz])
    intsy = np.concatenate([intsy * h2_frac, intsy])
    g = group_signals(build_spectrum(mz, intsy), noise=NoiseThreshold(min_rel=0.01))
    parent_mono = mz[len(mz) // 2]
    assert g.mono_mz(g.group_containing(parent_mono)) == pytest.approx(parent_mono)


# -- Multiply charged envelopes with sparse fractional evidence ----------------

CHARGED_DATA = Path(__file__).parent / "data" / "multiply_charged.mgf"


def test_real_multiply_charged_envelopes():
    """Real spectra read as z=1: a 3+ (beyond the old max_charge=2), and two 2+
    whose only fractional peak is M+1 (M+3 too weak)."""
    expected = {273.88268: 3, 218.68655: 2, 601.28538: 2}
    for spec in read_mgf(CHARGED_DATA):
        precursor = float(spec.metadata['PEPMASS'].split()[0])
        g = group_signals(spec.spec_arr, noise=NoiseThreshold(min_rel=0.05))
        assert int(g.charge[g.group_containing(precursor)]) == expected[precursor]


def test_2plus_with_only_fractional_m1():
    z2 = ISOTOPE_SPACING / 2
    mz = [500.0, 500.0 + z2, 500.0 + 2 * z2]   # M0, M+1, M+2 of a 2+; no M+3
    g = group_signals(build_spectrum(mz, [1.0, 0.6, 0.2]))
    assert int(g.charge[g.group_containing(500.0)]) == 2
