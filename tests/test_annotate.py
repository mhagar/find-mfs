"""
Tests for annotate_precursor(): precursor -> ranked (formula, adduct).

Exercises the paths MzKit depends on:
  1. count constraints defining the element set
  2. halogen_cap widening the search from the MS1 isotope envelope
  3. the MS2 term (MistNet) reranking toward the true formula

Real data: thiamphenicol (C12H15Cl2NO5S), a dichlorinated antibiotic, in
tests/data/thiamphenicol.mgf (extracted from test_spectra.mgf). The block is an
MS1 scan with in-source fragmentation + Na/K adducts, so one spectrum yields both
the [M+H]+ Cl2 envelope AND usable fragments. Note PEPMASS=338 is [M+H-H2O]+;
the intact [M+H]+ (356.0126) is the base peak inside the spectrum.
"""
from pathlib import Path

import pytest
from molmass import Formula

from find_mfs import annotate_precursor, FormulaScorer
from find_mfs.annotate import resolve_search_bounds
from find_mfs.spectra import read_mgf, to_spec_arr
from find_mfs.spectra.envelopes import SpectrumArray
from find_mfs.ms2.net import bundled_npz_path

DATA = Path(__file__).parent / "data" / "thiamphenicol.mgf"
INF = float("inf")
MH_PRECURSOR = 356.0126                     # [M+H]+ of C12H15Cl2NO5S (~1.5 ppm)
TRUE = Formula("C12H15Cl2NO5S").formula     # canonical Hill string


def _canon(s: str) -> str:
    return Formula(s).formula


def _rank_of_true(results) -> int | None:
    """0-based rank of the true formula in a sorted results set, else None."""
    for i, c in enumerate(results.candidates):
        if _canon(c.formula.formula) == TRUE:
            return i
    return None


def _crop(spec: SpectrumArray, lo: float, hi: float) -> SpectrumArray:
    return spec[(spec["mz"] >= lo) & (spec["mz"] <= hi)]


@pytest.fixture(scope="module")
def spectrum() -> SpectrumArray:
    specs = read_mgf(DATA)
    assert len(specs) == 1
    return specs[0].spec_arr


# --- count constraints define the element set ------------------------------

def test_resolve_bounds_derives_elements_from_max_counts():
    elements, max_b, min_b = resolve_search_bounds(
        "C*H*N*O*P0S2", "C1", halogen_cap=None, halogenated=False,
    )
    assert elements == "CHNOS"                       # P0 -> P left out entirely
    assert max_b == {"C": INF, "H": INF, "N": INF, "O": INF, "S": 2}
    assert min_b == {"C": 1, "H": 0, "N": 0, "O": 0, "S": 0}


def test_resolve_bounds_halogen_cap_overrides_max_counts():
    """When detection fires, the cap replaces the user's Cl/Br bounds -- even an
    explicit Cl0 -- and widens the element set."""
    elements, max_b, _ = resolve_search_bounds(
        "C*H*N*O*Cl0", None, halogen_cap="Cl4Br3", halogenated=True,
    )
    assert elements == "CHNOClBr"
    assert max_b["Cl"] == 4 and max_b["Br"] == 3


def test_resolve_bounds_cap_unused_when_not_halogenated():
    elements, max_b, _ = resolve_search_bounds(
        "C*H*N*O*Cl1", None, halogen_cap="Cl4Br3", halogenated=False,
    )
    assert elements == "CHNOCl"
    assert max_b["Cl"] == 1


@pytest.mark.parametrize("kwargs, match", [
    (dict(max_counts="C*H*", halogen_cap="F2"), "halogen_cap"),
    (dict(max_counts="C*H*", halogen_cap=""), "halogen_cap"),
    (dict(max_counts="C0H0"), "allows no elements"),
    (dict(max_counts="C*H*O*", min_counts="N1"), "min_counts requires"),
])
def test_resolve_bounds_rejects_bad_constraints(kwargs, match):
    kwargs = {"min_counts": None, "halogen_cap": None, **kwargs}
    with pytest.raises(ValueError, match=match):
        resolve_search_bounds(halogenated=True, **kwargs)


def test_counts_passed_via_finder_kwargs_are_rejected():
    with pytest.raises(TypeError, match="directly"):
        annotate_precursor(
            MH_PRECURSOR, adducts="H", finder_kwargs={"max_counts": "C*H*"},
        )


def test_max_counts_alone_reaches_halogen_formula(spectrum):
    """No separate element set: naming Cl in max_counts is enough to search it."""
    hits = annotate_precursor(
        MH_PRECURSOR, adducts="H", max_counts="C*H*N*O*S*Cl*", error_ppm=5.0,
        ms1_peaks=_crop(spectrum, 355.5, 361.0),
    )
    assert _rank_of_true(hits) is not None
    assert hits.query_params["elements"] == "CHNOSCl"
    assert hits.query_params["halogen_detected"] is None   # detection not asked for


# --- halogen_cap --------------------------------------------------------------

def test_halogen_cap_widens_constrained_search(spectrum):
    """The [M+H]+ Cl2 envelope must widen a halogen-free max_counts, so the true
    dichlorinated formula becomes reachable. This is the case that used to be a
    silent no-op: widening the element set while max_counts capped Cl/Br at 0."""
    hits = annotate_precursor(
        MH_PRECURSOR, adducts="H", max_counts="C*H*N*O*P0S2", error_ppm=5.0,
        halogen_cap="Cl4Br3", ms1_peaks=_crop(spectrum, 355.5, 361.0),
    )
    assert hits.query_params["halogen_detected"] is True
    assert hits.query_params["max_counts"]["Cl"] == 4
    assert _rank_of_true(hits) is not None, "true Cl2 formula not among candidates"


def test_halogen_cap_is_a_real_cap(spectrum):
    """A cap below the true Cl count must exclude the true formula."""
    hits = annotate_precursor(
        MH_PRECURSOR, adducts="H", max_counts="C*H*N*O*P0S2", error_ppm=5.0,
        halogen_cap="Cl1Br1", ms1_peaks=_crop(spectrum, 355.5, 361.0),
    )
    assert hits.query_params["halogen_detected"] is True
    assert _rank_of_true(hits) is None


def test_no_halogen_cap_cannot_reach_halogen_formula(spectrum):
    """Control: CHNOPS-only decomposition can't contain Cl, so the true formula
    is absent -- proving the hit above came from widening, not coincidence."""
    hits = annotate_precursor(
        MH_PRECURSOR, adducts="H", error_ppm=5.0,
        ms1_peaks=_crop(spectrum, 355.5, 361.0),
    )
    assert hits.query_params["halogen_detected"] is None
    assert _rank_of_true(hits) is None
    assert all("Cl" not in c.formula.formula for c in hits.candidates)


def test_halogen_cap_noop_on_clean_envelope():
    """A monoisotopic-only envelope (no M+2 zig-zag) must NOT widen."""
    clean = to_spec_arr([356.0126, 357.016], [1.0, 0.13])
    hits = annotate_precursor(
        MH_PRECURSOR, adducts="H", error_ppm=5.0,
        halogen_cap="Cl4Br3", ms1_peaks=clean,
    )
    assert hits.query_params["halogen_detected"] is False
    assert hits.query_params["elements"] == "CHNOPS"
    assert all("Cl" not in c.formula.formula for c in hits.candidates)


# --- MS2 reranking ----------------------------------------------------------

ARTIFACT = bundled_npz_path()


@pytest.mark.skipif(not ARTIFACT.exists(), reason="needs bundled MistNet npz")
def test_ms2_reranks_toward_true_formula(spectrum):
    """MS2 must not push the true formula down, and should surface it near top.
    Fragments (< precursor) drive MistNet; halogens in the element set so the
    true formula is a candidate."""
    common = dict(
        adducts="H", max_counts="C*H*N*O*P*S*F*Cl*Br*I*", error_ppm=5.0,
        ms1_peaks=_crop(spectrum, 355.5, 361.0),
        ms2_peaks=_crop(spectrum, 50.0, 355.0),
        scorer=FormulaScorer().with_ms2(ARTIFACT),
    )
    on = annotate_precursor(MH_PRECURSOR, ms2_weight=5.0, **common)
    off = annotate_precursor(MH_PRECURSOR, ms2_weight=0.0, **common)

    r_on, r_off = _rank_of_true(on), _rank_of_true(off)
    assert r_on is not None and r_off is not None
    assert r_on <= r_off, f"MS2 worsened the true formula's rank ({r_off} -> {r_on})"
    assert r_on < 5, f"true formula not near top with MS2 (rank {r_on})"
