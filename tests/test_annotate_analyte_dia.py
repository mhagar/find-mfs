"""
End-to-end tests for annotate_analyte_dia(): MS1 scan + precursor -> ranked mf.

Real data: thiamphenicol (C12H15Cl2NO5S) in tests/data/thiamphenicol.mgf. The base
peak is the water-loss [M+H-H2O]+ (338); the intact [M+H]+ (356.0126) is present
too, so the MDN must link them and resolve the precursor as [M+H]+.
"""
from pathlib import Path

import numpy as np
import pytest
from molmass import Formula

from find_mfs import annotate_analyte_dia
from find_mfs.spectra import read_mgf
from find_mfs.spectra.grouping import NoiseThreshold

DATA = Path(__file__).parent / "data" / "thiamphenicol.mgf"
MH_PRECURSOR = 356.0126
TRUE = Formula("C12H15Cl2NO5S").formula


def _canon(s: str) -> str:
    return Formula(s).formula


def _rank_of_true(results) -> int | None:
    for i, c in enumerate(results.candidates):
        if _canon(c.formula.formula) == TRUE:
            return i
    return None


@pytest.fixture(scope="module")
def spectrum():
    specs = read_mgf(DATA)
    assert len(specs) == 1
    return specs[0].spec_arr


def test_auto_selects_tallest_envelope_when_no_precursor(spectrum):
    """DIA default: no precursor -> base is the tallest signal (here the water-loss
    [M+H-H2O]+ at 338), and the MDN still recovers the correct neutral M and the
    true formula. The used precursor is reported back."""
    res = annotate_analyte_dia(spectrum, error_ppm=8.0, detect_halogens=True)
    assert abs(res.precursor_mz - 338.0023) < 0.01     # base peak = water loss
    assert res.adduct == "-OH"
    assert res.is_halogen is True
    assert abs(res.M - 355.005) < 0.01                 # same analyte mass as [M+H]+
    assert _rank_of_true(res.candidates) is not None


def test_auto_does_not_annotate_the_fragment_as_analyte(spectrum):
    """The base peak here is the in-source water-loss [M+H-H2O]+. By default only
    the winning adduct is decomposed, so the intact formula wins -- the water-loss
    fragment formula must NOT be offered as a competing analyte."""
    res = annotate_analyte_dia(spectrum, error_ppm=8.0, detect_halogens=True)
    assert res.adduct == "-OH"
    assert _canon(res.candidates[0].formula.formula) == TRUE
    # Near-ties are reported as metadata, not merged into candidates by default.
    assert res.near_tie_adducts is not None
    assert all(c.adduct == "-OH" for c in res.candidates)


def test_explicit_and_auto_agree_on_analyte_mass(spectrum):
    auto = annotate_analyte_dia(spectrum, error_ppm=8.0)
    expl = annotate_analyte_dia(spectrum, precursor_mz=MH_PRECURSOR, error_ppm=8.0)
    assert abs(auto.M - expl.M) < 0.01
    assert expl.precursor_mz == MH_PRECURSOR


def test_resolves_precursor_and_finds_true_formula(spectrum):
    # Explicitly forcing the *minor* intact envelope (356, ~25% intensity): its weak
    # M+1 isotope peak sits below the default 5% floor, so a lower noise floor is
    # needed to keep the envelope's fine structure for halogen detection.
    res = annotate_analyte_dia(
        spectrum, precursor_mz=MH_PRECURSOR, elements="CHNOPS", error_ppm=8.0,
        detect_halogens=True, noise=NoiseThreshold(min_rel=0.005),
    )
    assert res.adduct == "H"
    assert res.charge == 1
    assert res.is_halogen is True
    # Neutral mass = precursor - proton (~355.005 for C12H15Cl2NO5S).
    assert abs(res.M - 355.005) < 0.01
    r = _rank_of_true(res.candidates)
    assert r is not None, "true dichlorinated formula not among candidates"


def test_halogen_widening_is_required(spectrum):
    """Without halogen detection the element set stays CHNOPS -> no Cl formula."""
    res = annotate_analyte_dia(
        spectrum, precursor_mz=MH_PRECURSOR, elements="CHNOPS", error_ppm=8.0,
        detect_halogens=False,
    )
    assert _rank_of_true(res.candidates) is None
    assert all("Cl" not in c.formula.formula for c in res.candidates)


def test_grouped_is_populated_for_viz(spectrum):
    res = annotate_analyte_dia(
        spectrum, precursor_mz=MH_PRECURSOR, error_ppm=8.0, detect_halogens=True,
    )
    g = res.grouped
    base = res.base_group_id
    # The base group's labels round-trip and were filled by the MDN.
    assert g.adduct_label[base] == "[M+H]+"
    assert not np.isnan(g.M[base])
    assert len(g.peaks_of(base)) >= 1


def test_missing_precursor_raises(spectrum):
    with pytest.raises(ValueError):
        annotate_analyte_dia(spectrum, precursor_mz=9999.0, precursor_tol=0.01)
