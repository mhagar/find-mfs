"""
Tests for charge-aware signal grouping (struct-of-arrays).

Covers charge inference (1+ vs 2+ from isotope spacing), singletons, the three
noise-threshold modes, and halogen flagging on real Cl2 data.
"""
from pathlib import Path

import numpy as np

from find_mfs.spectra import read_mgf
from find_mfs.spectra.envelopes import build_spectrum, ISOTOPE_SPACING
from find_mfs.spectra.grouping import group_signals, NoiseThreshold

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
