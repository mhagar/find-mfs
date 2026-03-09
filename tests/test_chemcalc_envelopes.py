"""
End-to-end tests using reference envelopes from ChemCalc
(Should replace these with envelopes from real experimental data at some point)

For each compound × adduct combination, we feed the monoisotopic m/z
into FormulaFinder with the ChemCalc envelope as observed data, and
verify that the correct neutral formula is among the results with a
good isotope match.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from molmass import Formula

from find_mfs.core.finder import FormulaFinder, FormulaCandidate
from find_mfs.isotopes.config import SingleEnvelopeMatch
from find_mfs.utils import formula_match


DATA_FILE = Path(__file__).parent / "data" / "chemcalc_envelopes.json"

with open(DATA_FILE) as f:
    _DATA = json.load(f)


def _build_test_cases():
    """
    Yield (test_id, neutral_formula, adduct_str, charge, envelope_array)
    """
    adduct_map = {
        "[M+H]+":  ("H",   1),
        "[M+Na]+": ("Na",  1),
        "[M-H]-":  ("-H", -1),
        "[M+Cl]-": ("Cl", -1),
    }
    for compound in _DATA["compounds"]:
        name = compound["name"]
        neutral = compound["neutral_formula"]
        for entry in compound["adducts"]:
            adduct_label = entry["adduct"]
            adduct_str, charge = adduct_map[adduct_label]
            envelope = np.array(
                [[p["mz"], p["intensity"]] for p in entry["envelope"]]
            )
            test_id = f"{name}_{adduct_label}"
            yield pytest.param(
                neutral, adduct_str, charge, envelope,
                id=test_id,
            )


# Per-compound configuration: element set and optional count constraints.
# Large molecules (MW > ~800) need element constraints to keep the
# decomposition tractable with CHNOPS.
_COMPOUND_CONFIG = {
    "C31H36N2O11": {                   # Novobiocin (MW ~612)
        "elements": "CHNOPS",
    },
    "C12H15Cl2NO5S": {                 # Thiamphenicol (MW ~355)
        "elements": "CHNOPSCl",
    },
    "C62H111N11O12": {                 # Cyclosporine (MW ~1202)
        "elements": "CHNOPS",
        "max_counts": {"C": 70, "H": 130, "N": 15, "O": 15, "P": 2, "S": 5},
    },
}

# Cache finders by element set to avoid rebuilding ERT tables
_FINDERS: dict[str, FormulaFinder] = {}


def _get_finder(neutral_formula: str) -> FormulaFinder:
    elements = _COMPOUND_CONFIG[neutral_formula]["elements"]
    if elements not in _FINDERS:
        _FINDERS[elements] = FormulaFinder(elements)
    return _FINDERS[elements]


def _get_constraints(neutral_formula: str) -> dict:
    cfg = _COMPOUND_CONFIG[neutral_formula]
    kwargs = {}
    if "max_counts" in cfg:
        kwargs["max_counts"] = cfg["max_counts"]
    if "min_counts" in cfg:
        kwargs["min_counts"] = cfg["min_counts"]
    return kwargs


@pytest.mark.parametrize(
    "neutral_formula, adduct_str, charge, envelope",
    list(_build_test_cases()),
)
def test_chemcalc_envelope(
    neutral_formula: str,
    adduct_str: str,
    charge: int,
    envelope: np.ndarray,
):
    """
    End-to-end: given an observed m/z and isotope envelope from ChemCalc,
    the finder should return the correct neutral formula with a good
    isotope match score.
    """
    finder = _get_finder(neutral_formula)

    # The monoisotopic m/z is the first peak in the envelope
    mono_mz = envelope[0, 0]

    isotope_config = SingleEnvelopeMatch(
        envelope=envelope.copy(),
        mz_tolerance_da=0.01,
        minimum_rmse=0.10,
        enable_approx_prefilter=False,
    )

    constraints = _get_constraints(neutral_formula)

    results = finder.find_formulae(
        mass=mono_mz,
        charge=charge,
        adduct=adduct_str,
        error_ppm=5.0,
        isotope_match=isotope_config,
        **constraints,
    )

    target_formula = Formula(neutral_formula)

    # Find the expected formula among results
    matches = [
        r for r in results
        if formula_match(target_formula, r.formula)
    ]

    assert len(matches) == 1, (
        f"Expected exactly 1 match for {neutral_formula} "
        f"(adduct={adduct_str}, charge={charge}), "
        f"got {len(matches)}. "
        f"Total results: {len(results)}. "
        f"Formulae returned: {[r.formula.formula for r in results[:20]]}"
    )

    candidate = matches[0]

    # Mass error should be very small (exact mass from ChemCalc)
    assert abs(candidate.error_ppm) < 5.0, (
        f"{neutral_formula} [{adduct_str}]: "
        f"error_ppm={candidate.error_ppm:.4f}, expected < 5.0 ppm"
    )

    # Isotope match should be present and good
    assert candidate.isotope_match_result is not None, (
        f"{neutral_formula} [{adduct_str}]: no isotope match result"
    )
    assert candidate.isotope_match_result.intensity_rmse < 0.10, (
        f"{neutral_formula} [{adduct_str}]: "
        f"isotope RMSE={candidate.isotope_match_result.intensity_rmse:.4f}, "
        f"expected < 0.10"
    )
