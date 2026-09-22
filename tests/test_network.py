"""
Tests for the mass-difference network (candidate-M clustering).

The headline case is degenerate-ion resolution: a pair of signals separated by an
NH3 mass is an NH3 adduct or an NH3 loss depending on which signal carries a metal
anchor, and the two imply different neutral masses M. Also covers metal-free
resolution, charge-state co-clustering, multimers, and near-tie reporting.
"""
import numpy as np

from find_mfs.spectra.envelopes import build_spectrum, ISOTOPE_SPACING
from find_mfs.spectra.grouping import group_signals
from find_mfs.spectra.network import solve_for_base
from find_mfs.spectra.ions import _H, _NA, _NH3, _NH4, _H2O, ELECTRON_MASS

M = 300.0


def _mh(m):   return m + _H - ELECTRON_MASS          # [M+H]+
def _mna(m):  return m + _NA - ELECTRON_MASS         # [M+Na]+
def _mnh4(m): return m + _NH4 - ELECTRON_MASS        # [M+NH4]+
def _mhh2o(m): return m + _H - _H2O - ELECTRON_MASS  # [M+H-H2O]+


def _solve(mzs, intsys, base_mz):
    g = group_signals(build_spectrum(mzs, intsys))
    base = g.group_containing(base_mz)
    return g, solve_for_base(g, base)


def test_degenerate_nh3_resolves_as_adduct_with_metal_on_lighter():
    """Na partner on the lighter signal A => A=[M+H]+, B=[M+NH4]+ (one M=300)."""
    A = _mh(M)
    B = A + _NH3
    g, sol = _solve([A, B, _mna(M)], [1.0, 0.5, 0.6], base_mz=A)
    assert sol.base_ion.label == "[M+H]+"
    assert abs(sol.M - M) < 1e-3


def test_degenerate_nh3_resolves_as_loss_with_metal_on_heavier():
    """Na partner on the heavier signal B => B=[M+H]+, A=[M+H-NH3]+ (M shifts up)."""
    A = _mh(M)
    B = A + _NH3
    g, sol = _solve([A, B, _mna(M + _NH3)], [1.0, 0.5, 0.6], base_mz=A)
    assert sol.base_ion.label == "[M+H-NH3]+"
    assert abs(sol.M - (M + _NH3)) < 1e-3


def test_metal_free_resolution():
    """No metal anchor: [M+H]+, [M+H-H2O]+, [M+NH4]+ still pin one M."""
    A = _mh(M)
    g, sol = _solve(
        [A, _mhh2o(M), _mnh4(M)], [1.0, 0.7, 0.4], base_mz=A
    )
    assert sol.base_ion.label == "[M+H]+"
    assert abs(sol.M - M) < 1e-3
    # All three groups explained by the same M.
    assert len(sol.assignments) == 3
    assert all(abs(v - M) < 1e-3 for v in g.M[list(sol.assignments)])


def test_charge_states_co_cluster():
    """[M+H]+ and [M+2H]2+ of the same analyte resolve to one M / one component."""
    A = _mh(M)
    # 2+ envelope needs a real fractional (0.5 Da) ladder to be inferred as 2+.
    z2 = ISOTOPE_SPACING / 2
    mh2 = (M + 2 * _H - 2 * ELECTRON_MASS) / 2
    g, sol = _solve(
        [A, mh2, mh2 + z2, mh2 + 2 * z2, mh2 + 3 * z2],
        [1.0, 0.6, 0.35, 0.18, 0.07],
        base_mz=A,
    )
    assert sol.base_ion.label == "[M+H]+"
    twoplus = g.group_containing(mh2)
    assert twoplus in sol.assignments
    assert abs(g.M[twoplus] - M) < 1e-3


def test_dimer_supports_monomer_mass():
    """A [2M+H]+ dimer resolves to the same M as the monomer [M+H]+."""
    A = _mh(M)
    dimer = 2 * M + _H - ELECTRON_MASS
    g, sol = _solve([A, dimer], [1.0, 0.4], base_mz=A)
    assert sol.base_ion.label == "[M+H]+"
    dim_gid = g.group_containing(dimer)
    assert dim_gid in sol.assignments
    assert sol.assignments[dim_gid].label == "[2M+H]+"


def test_lone_ambiguous_pair_returns_near_ties():
    """A lone A/B pair (NH3 apart), no anchor: winner + the loss reading returned.
    Equal intensities keep the ambiguity genuine under intensity weighting."""
    A = _mh(M)
    B = A + _NH3
    g, sol = _solve([A, B], [1.0, 1.0], base_mz=A)
    labels = {ion.label for ion in sol.near_tie_ions}
    assert sol.base_ion.label == "[M+H]+"
    assert "[M+H-NH3]+" in labels
    assert len(sol.near_tie_ions) >= 2


def test_stray_half_peak_resolves_as_MH_not_dimer():
    """Regression: a 1+ [M+H]+ with a stray +0.5 peak (+ an Na partner) used to be
    mislabeled [2M+2H]2+. It must resolve as [M+H]+ with the correct M."""
    A = _mh(M)
    stray = A + ISOTOPE_SPACING / 2
    g, sol = _solve([A, A + ISOTOPE_SPACING, stray, _mna(M)],
                    [1.0, 0.3, 0.08, 0.6], base_mz=A)
    assert sol.base_ion.label == "[M+H]+"
    assert abs(sol.M - M) < 1e-3


def test_low_abundance_network_does_not_override_base():
    """A dominant [M+H]+ must not be reinterpreted just because a pair of trace
    peaks forms a richer-looking network at a shifted M."""
    base = _mh(M)                       # 100% [M+H]+
    Mp = M + _H2O                       # a shifted mass a trace network could imply
    trace1 = _mh(Mp)                    # [M'+H]+  (trace)
    trace2 = _mna(Mp)                   # [M'+Na]+ (trace)
    spec = build_spectrum([base, trace1, trace2], [1.0, 0.001, 0.001])
    g = group_signals(spec)
    b = g.group_containing(base)

    sol = solve_for_base(g, b)                          # intensity-weighted (default)
    assert sol.base_ion.label == "[M+H]+"
    assert abs(sol.M - M) < 1e-3

    # Intensity-blind reproduces the old failure: the trace pair wins.
    sol0 = solve_for_base(g, b, intensity_weight=0.0, write_back=False)
    assert sol0.base_ion.label != "[M+H]+"


def test_write_back_populates_grouped():
    A = _mh(M)
    g, sol = _solve([A, _mna(M)], [1.0, 0.5], base_mz=A)
    base = sol.base_gid
    assert g.adduct_label[base] == "[M+H]+"
    assert not np.isnan(g.M[base])
    assert g.component_id[base] == base
