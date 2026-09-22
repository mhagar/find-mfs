"""
Ion-type vocabulary for the mass-difference network (MDN).

This module is the single source of truth for what ion species the adduct-
determination step considers, and the knob you tune: comment an entry out of
`ION_VOCAB` to stop the MDN from ever proposing it.

## The mass model

Each observed signal is an ion. We deconvolute charge up front into a
charge-independent **deconvoluted mass**::

    D = z * mz + z * m_e

`D` is the neutral-equivalent mass of the *ion* (the `z * m_e` term adds the
electrons back that ionization stripped, which matters at the sub-ppm accuracy
modern instruments reach). Each ion type carries a multiplicity `n` (1 = monomer,
2 = dimer, ...) and an `offset` = the neutral mass of the atoms added to (or lost
from) the analyte. The analyte's neutral mass is then::

    M = (D - offset) / n

with charge captured in `D`, adduct composition in `offset`, and multimer in `n`
-- three independent axes. E.g. for [M+2H]2+, `D` already folds in the charge, so
its offset is just `2 * m_H` and `n = 1`; for [2M+H]+, offset is `m_H` and `n = 2`.

`adduct_str` / `charge` are what a candidate is handed to
`FormulaFinder.find_formulae` as (via `annotate_precursor`). Note the finder does
not convert m/z to neutral mass by charge, so only `charge == 1` ion types can be
passed a raw precursor m/z directly; multiply-charged base precursors are handled
by reconstructing a z=1-equivalent m/z from the resolved M (see `annotate.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

from molmass import Formula
from molmass.elements import ELECTRON

# Electron mass from molmass, to stay consistent with the molmass monoisotopic
# masses used for every offset below (do not swap for ms2.tables.ELECTRON_MASS,
# which is a frozen rdkit value kept only for the MistNet input contract).
ELECTRON_MASS = ELECTRON.mass


def _mass(formula_str: str) -> float:
    """Monoisotopic mass of a neutral formula fragment."""
    return Formula(formula_str).monoisotopic_mass


def deconv_mass(mz: float, charge: int) -> float:
    """
    Charge-independent deconvoluted mass of an ion: D = z*mz + z*m_e.

    Works elementwise on numpy arrays too.
    """
    return charge * mz + charge * ELECTRON_MASS


@dataclass(frozen=True, slots=True)
class IonType:
    """
    One ion species the MDN can propose.

    Attributes:
        label: Human-readable ion string, e.g. "[M+H]+", "[2M+Na]+".
        n: Multiplicity (1 = monomer, 2 = dimer).
        offset: Neutral mass of the added (+) / lost (-) atoms.
        charge: Absolute charge state (positive mode only for now).
        prior: Plausibility weight in [0, 1]; higher = more expected. Drives the
            parsimony scoring in the network. First-cut values, tune vs benchmark.
        adduct_str: Neutral adduct formula for FormulaFinder ("H", "Na", "-OH",
            ...), or None for a radical/no-adduct ion. Losses use a leading '-'.
    """
    label: str
    n: int
    offset: float
    charge: int
    prior: float
    adduct_str: str | None

    def neutral_mass(self, D: float) -> float:
        """Analyte neutral mass implied if a signal with deconv mass D is this ion."""
        return (D - self.offset) / self.n


# Reusable neutral-atom masses.
_H = _mass("H")
_NA = _mass("Na")
_K = _mass("K")
_NH4 = _mass("NH4")       # N + 4H
_NH3 = _mass("NH3")
_H2O = _mass("H2O")

# Ordered vocabulary. THIS IS THE TUNING KNOB: comment a line out to disallow that
# ion type. Priors: protonation/metals high, ammonium/water-loss moderate, radical
# and multimers low. Monomers first, then n=2 multimers.
ION_VOCAB: list[IonType] = [
    IonType("[M+H]+",      1, _H,          1, 1.00, "H"),
    IonType("[M+Na]+",     1, _NA,         1, 0.90, "Na"),
    IonType("[M+K]+",      1, _K,          1, 0.70, "K"),
    IonType("[M+NH4]+",    1, _NH4,        1, 0.50, "NH4"),
    IonType("[M+H-H2O]+",  1, _H - _H2O,   1, 0.50, "-OH"),
    IonType("[M+H-NH3]+",  1, _H - _NH3,   1, 0.45, "-NH2"),
    IonType("[M]+",        1, 0.0,         1, 0.20, None),
    IonType("[M+2H]2+",    1, 2 * _H,      2, 0.40, "H"),
    # --- multimers (n=2) --------------------------------------------------------
    IonType("[2M+H]+",     2, _H,          1, 0.40, "H"),
    IonType("[2M+Na]+",    2, _NA,         1, 0.35, "Na"),
    IonType("[2M+2H]2+",   2, 2 * _H,      2, 0.20, "H"),
]
