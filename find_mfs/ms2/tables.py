"""
Element / ion / instrument vocabularies for MistNet featurization

A torch-free, rdkit-free port of the pieces of `mist_cf.common.chem_utils`
needed for inference

Everything here defines the input contract of a trained checkpoint, so none of it
 may drift

## Why element masses are hardcoded rather than looked up:
The reference used rdkit's `GetMostCommonIsotopeMass`, which differs from molmass by up to 1.9e-6 Da
This is small but it feeds the CLS precursor-mass-error feature,
 and for now I'm more interested in reproducing the trained featurization exactly
"""

from __future__ import annotations

import re

import numpy as np

CHEM_FORMULA_SIZE = r"([A-Z][a-z]*)([0-9]*)"
ELECTRON_MASS = 0.00054858

# Order defines the element-count vector layout. DO NOT REORDER.
VALID_ELEMENTS = [
    "C", "N", "P", "O", "S", "Si", "I", "H", "Cl",
    "F", "Br", "B", "Se", "Fe", "Co", "As", "K", "Na",
]
NUM_ELEMENTS = len(VALID_ELEMENTS)

# Monoisotopic masses as rdkit reported them at training time (see module docstring).
ELEMENT_TO_MASS = {
    "C": 12.0,
    "N": 14.003074,
    "P": 30.97376163,
    "O": 15.99491462,
    "S": 31.972071,
    "Si": 27.97692653,
    "I": 126.904473,
    "H": 1.007825032,
    "Cl": 34.96885268,
    "F": 18.99840322,
    "Br": 78.9183371,
    "B": 11.0093054,
    "Se": 79.9165213,
    "Fe": 55.9349375,
    "Co": 58.933195,
    "As": 74.9215965,
    "K": 38.96370668,
    "Na": 22.98976928,
}

_ELEMENT_TO_IDX = {e: i for i, e in enumerate(VALID_ELEMENTS)}
_ELEMENT_VECTORS = np.eye(NUM_ELEMENTS)

# The two element sets the MS2 stage searches. Both are find-mfs pre-calculated
# ERTs, so either loads instantly. `use_halogens` throughout this package indexes
# into this map -- resolve it here rather than hardcoding element sets at call sites.
ELEMENTS_FOR = {False: "CHNOPS", True: "CHNOPSFClBrI"}

# --- ions -------------------------------------------------------------------- #

ION_LST = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",
    "[M]+",
    "[M-H4O2+H]+",
]
NUM_ION = len(ION_LST)
_ION_TO_IDX = {ion: i for i, ion in enumerate(ION_LST)}

# Twin-blind collapse: the CHNOPS-degenerate adducts share identical MS2 evidence
# AND the same precursor mass, so MS2 cannot separate them even in principle.
# Collapsing them to one one-hot column forces equal logits across the twins; an
# MS1 adduct-partner term breaks the tie downstream. Na/K stay distinct because
# they ARE mass-distinguishable. Gated by the checkpoint's `collapse_ions` flag.
ION_CORE_TWINS = ("[M+H]+", "[M-H2O+H]+", "[M+H3N+H]+", "[M]+", "[M-H4O2+H]+")
ION_CLASSES = ["core", "[M+Na]+", "[M+K]+"]
NUM_ION_CLASSES = len(ION_CLASSES)
_ION_CLASS_TO_IDX = {c: i for i, c in enumerate(ION_CLASSES)}
ION_TO_COLLAPSED_IDX = {
    ion: _ION_CLASS_TO_IDX["core" if ion in ION_CORE_TWINS else ion] for ion in ION_LST
}

# MIST-CF ion string -> (find-mfs net-neutral adduct, charge).
# find-mfs subtracts the adduct (and the electron mass, via `charge`) before
# decomposing, and reports the *neutral core* formula -- which is exactly what
# MistNet scores. Losses are pre-netted into a single subtraction formula
# (find-mfs accepts one leading '-'), e.g. [M-H2O+H]+ == net "-OH".
ION_TO_ADDUCT = {
    "[M+H]+": ("H", 1),
    "[M+Na]+": ("Na", 1),
    "[M+K]+": ("K", 1),
    "[M-H2O+H]+": ("-OH", 1),
    "[M+H3N+H]+": ("NH4", 1),
    "[M]+": (None, 1),
    "[M-H4O2+H]+": ("-H3O2", 1),
}

ION_TO_MASS = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
}

# --- instruments ------------------------------------------------------------- #

INSTRUMENT_TO_TYPE = {
    "Thermo Finnigan Velos Orbitrap": "orbitrap",
    "Thermo Finnigan Elite Orbitrap": "orbitrap",
    "Orbitrap Fusion Lumos": "orbitrap",
    "Q-ToF (LCMS)": "qtof",
    "Unknown (LCMS)": "unknown",
    "Ion Trap (LCMS)": "iontrap",
    "ion trap": "iontrap",
    "FTICR (LCMS)": "fticr",
    "Bruker Q-ToF (LCMS)": "qtof",
    "Orbitrap (LCMS)": "orbitrap",
    "ESI-Orbitrap": "orbitrap",
    "ESI-qTOF": "qtof",
    "ESI-qToF": "qtof",
    "ESI-qTof": "qtof",
    # Canonical names map to themselves, so datasets already storing the
    # canonical instrument work unchanged.
    "qtof": "qtof",
    "orbitrap": "orbitrap",
    "iontrap": "iontrap",
    "fticr": "fticr",
    "unknown": "unknown",
    "nan": "unknown",
}
INSTRUMENTS = sorted(set(INSTRUMENT_TO_TYPE.values()))
# One spare column beyond the known instruments, as the reference allocates.
MAX_INSTR_IDX = len(INSTRUMENTS) + 1
_INSTRUMENT_TO_IDX = {name: i for i, name in enumerate(INSTRUMENTS)}


def get_ion_idx(ion: str, collapse: bool = False) -> int:
    """One-hot column for ``ion``; ``collapse`` selects the twin-blind head."""
    return ION_TO_COLLAPSED_IDX[ion] if collapse else _ION_TO_IDX[ion]


def get_instr_idx(instrument: str) -> int:
    """One-hot column for ``instrument``; unknown names fall back to 'unknown'."""
    return _INSTRUMENT_TO_IDX[INSTRUMENT_TO_TYPE.get(instrument, "unknown")]


# --- formulae ---------------------------------------------------------------- #


def formula_to_dense(
        chem_formula: str
) -> np.ndarray:
    """
    Hill-string formula -> `(NUM_ELEMENTS,)` element-count vector.

    Raises KeyError on elements outside :data:`VALID_ELEMENTS`, matching the
    reference (a silent zero would be a worse failure).
    """
    total = []
    for symbol, num in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        count = 1 if num == "" else int(num)
        total.append(np.repeat(_ELEMENT_VECTORS[_ELEMENT_TO_IDX[symbol]][None, :], count, axis=0))
    if not total:
        return np.zeros(NUM_ELEMENTS)
    return np.vstack(total).sum(0)


def formula_mass(chem_formula: str) -> float:
    """Monoisotopic mass of a neutral Hill-string formula."""
    mass = 0.0
    for symbol, num in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        mass += ELEMENT_TO_MASS[symbol] * (1 if num == "" else int(num))
    return mass


def electron_correct(mass: float) -> float:
    return mass - ELECTRON_MASS


def get_cls_mass_diff(parentmass: float, form: str, ion: str, corr_electrons: bool = True) -> float:
    """|observed precursor m/z - theoretical m/z of (form, ion)|."""
    true_val = formula_mass(form) + ION_TO_MASS[ion]
    if corr_electrons:
        true_val = electron_correct(true_val)
    return abs(parentmass - true_val)


def norm_mass_diff_ppm(mass_diff):
    """The network's ppm scaling."""
    return mass_diff / 10


def clipped_ppm_single(mass_diff: float, parentmass: float) -> float:
    """ppm error, with the denominator floored at 200 Da (small-mass guard)."""
    return mass_diff / (200 if parentmass < 200 else parentmass) * 1e6


def clipped_ppm_single_norm(mass_diff: float, parentmass: float) -> float:
    return norm_mass_diff_ppm(clipped_ppm_single(mass_diff, parentmass))
