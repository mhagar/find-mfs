"""
Adduct pair delta computation, adduct grouping, and adduct identification
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from molmass import Formula

from .envelopes import SpectrumArray, _envelope_stats


# -- Adduct pair deltas --------------------------------------------------------
def build_adduct_pair_deltas(
    adduct_specs: list[tuple[str, str, int]],
) -> dict[tuple[str, str], float]:
    """
    Compute {(lighter_label, heavier_label): delta_mz} for all unique pairs.

    adduct_specs: list of (label, formula_str, sign)
      sign +1 = adduct is added to M, -1 = adduct is lost from M.
    Pairs are stored with positive delta (lighter adduct first).

    For neutral losses (sign=-1, label not starting with '-'), the pair delta
    against any other entry is just the loss mass — because the loss modifies
    whatever base ion is present, e.g. [M+H-H2O]+ is 18.01 Da below [M+H]+.
    """
    masses = {
        label: Formula(formula_str).monoisotopic_mass
        for label, formula_str, sign in adduct_specs
    }
    signs = {label: sign for label, formula_str, sign in adduct_specs}
    loss_labels = {
        label for label, formula_str, sign in adduct_specs
        if sign == -1 and not label.startswith('-')
    }

    offsets = {
        label: signs[label] * masses[label]
        for label in masses
    }

    pairs: dict[tuple[str, str], float] = {}
    labels = list(offsets)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            a_is_loss = a in loss_labels
            b_is_loss = b in loss_labels

            if a_is_loss and b_is_loss:
                # Both losses: delta is difference in loss masses
                delta = abs(masses[a] - masses[b])
            elif a_is_loss or b_is_loss:
                # One loss, one adduct: delta is just the loss mass
                delta = masses[a] if a_is_loss else masses[b]
            else:
                # Both adducts: use offset difference
                delta = abs(offsets[b] - offsets[a])

            # Store with the lighter m/z label first
            if a_is_loss and not b_is_loss:
                # Loss product is lighter than its base ion
                pairs[(a, b)] = delta
            elif b_is_loss and not a_is_loss:
                pairs[(b, a)] = delta
            elif offsets.get(a, 0) <= offsets.get(b, 0):
                pairs[(a, b)] = delta
            else:
                pairs[(b, a)] = delta

    return pairs


POSITIVE_ADDUCT_SPECS: list[tuple[str, str, int]] = [
    ("H",   "H",   +1),
    ("Na",  "Na",  +1),
    ("H2O", "H2O", -1),  # Far more likely of being ISF than adduct
    ("NH4", "NH4", -1),  # Far more likely of being ISF than adduct
    # ("H2", "H2", -1),
    # ("K",   "K",   +1),
    # ("NH4", "NH4", +1),
]

NEGATIVE_ADDUCT_SPECS: list[tuple[str, str, int]] = [
    ("-H",   "H",    -1),
    ("Cl",   "Cl",   +1),
    ("HCOO", "HCOOH", +1),
]

def get_loss_labels(
    adduct_specs: list[tuple[str, str, int]],
) -> frozenset[str]:
    """Labels that represent neutral losses (sign=-1, excluding deprotonation)."""
    return frozenset(
        label for label, formula_str, sign in adduct_specs
        if sign == -1 and not label.startswith('-')
    )


POSITIVE_PAIR_DELTAS = build_adduct_pair_deltas(POSITIVE_ADDUCT_SPECS)
NEGATIVE_PAIR_DELTAS = build_adduct_pair_deltas(NEGATIVE_ADDUCT_SPECS)
POSITIVE_LOSS_LABELS = get_loss_labels(POSITIVE_ADDUCT_SPECS)
NEGATIVE_LOSS_LABELS = get_loss_labels(NEGATIVE_ADDUCT_SPECS)


# -- Adduct grouping -----------------------------------------------------------

def find_adduct_groups(
    spec_arr: SpectrumArray,
    envelope_labels: NDArray[np.intp],
    adduct_pair_deltas: dict[tuple[str, str], float],
    tol_da: float = 0.01,
    tol_ppm: float = 5.0,
) -> NDArray[np.intp]:
    """
    Greedy adduct group detection. Processes envelopes tallest-first.

    Tolerance per comparison: max(tol_da, seed_mz * tol_ppm * 1e-6).

    Returns adduct_labels, shape (n_envelopes,).
    adduct_labels[i] = adduct group ID for envelope i.
    """
    n_envelopes = int(envelope_labels.max()) + 1
    mono_mz, max_intsy = _envelope_stats(spec_arr, envelope_labels, n_envelopes)

    adduct_labels = np.full(n_envelopes, -1, dtype=np.intp)
    intensity_order = np.argsort(max_intsy)[::-1]
    expected_deltas = np.fromiter(adduct_pair_deltas.values(), dtype=np.float64)

    group_id = 0
    for seed_eid in intensity_order:
        if adduct_labels[seed_eid] != -1:
            continue

        adduct_labels[seed_eid] = group_id
        seed_mz = mono_mz[seed_eid]
        tol = max(tol_da, seed_mz * tol_ppm * 1e-6)

        for other_eid in range(n_envelopes):
            if adduct_labels[other_eid] != -1:
                continue
            delta = abs(mono_mz[other_eid] - seed_mz)
            if np.any(np.abs(expected_deltas - delta) <= tol):
                adduct_labels[other_eid] = group_id

        group_id += 1

    return adduct_labels


# -- Adduct identification ----------------------------------------------------

def _net_adduct_string(
    base_label: str,
    loss_label: str,
    adduct_specs: list[tuple[str, str, int]],
) -> str | None:
    """
    Compute the net adduct string for a loss product.

    E.g. base="H" (sign +1) + loss="H2O" (sign -1) → net is +H -H2O = -OH.

    Returns a FormulaFinder-compatible adduct string ("-OH"), or None if the
    net composition has mixed positive/negative element counts.
    """
    specs_by_label = {label: (formula_str, sign) for label, formula_str, sign in adduct_specs}
    if base_label not in specs_by_label or loss_label not in specs_by_label:
        return None

    base_formula_str, base_sign = specs_by_label[base_label]
    loss_formula_str, loss_sign = specs_by_label[loss_label]

    base_f = Formula(base_formula_str)
    loss_f = Formula(loss_formula_str)

    # Net element counts: base_sign * base_composition + loss_sign * loss_composition
    net: dict[str, int] = {}
    for el, item in base_f.composition().items():
        net[el] = net.get(el, 0) + base_sign * item.count
    for el, item in loss_f.composition().items():
        net[el] = net.get(el, 0) + loss_sign * item.count

    # Remove zeros
    net = {el: c for el, c in net.items() if c != 0}
    if not net:
        return None

    all_positive = all(c > 0 for c in net.values())
    all_negative = all(c < 0 for c in net.values())

    if not (all_positive or all_negative):
        # Mixed signs — can't express as a simple adduct string
        return None

    # Build formula string (Hill order: C first, H second, then alphabetical)
    counts = {el: abs(c) for el, c in net.items()}
    parts = []
    for el in sorted(counts, key=lambda e: (e != 'C', e != 'H', e)):
        c = counts[el]
        parts.append(f"{el}{c}" if c > 1 else el)
    formula_str = ''.join(parts)

    return f"-{formula_str}" if all_negative else formula_str


def identify_adduct(
    seed_mz: float,
    partner_mzs: NDArray[np.float64],
    adduct_pair_deltas: dict[tuple[str, str], float],
    tol_da: float,
    tol_ppm: float,
    loss_labels: frozenset[str] = frozenset(),
    adduct_specs: list[tuple[str, str, int]] | None = None,
) -> str | None:
    """
    Return the adduct string for the seed envelope given partner mz values,
    or None if no confident match.

    delta_signed > 0 means partner is heavier → seed is the lighter adduct.

    If the resolved label is a neutral loss (in loss_labels), the net adduct
    is computed by combining the base adduct with the loss. E.g. an envelope
    18 Da below [M+H]+ gets adduct "-OH" (= +H - H2O).
    """
    tol = max(tol_da, seed_mz * tol_ppm * 1e-6)
    for partner_mz in partner_mzs:
        delta_signed = partner_mz - seed_mz
        abs_delta = abs(delta_signed)
        for (lighter, heavier), pair_delta in adduct_pair_deltas.items():
            if abs(abs_delta - pair_delta) <= tol:
                label = lighter if delta_signed > 0 else heavier
                if label in loss_labels and adduct_specs is not None:
                    # This envelope is a loss product; compute net adduct
                    base_label = heavier if delta_signed > 0 else lighter
                    net = _net_adduct_string(base_label, label, adduct_specs)
                    return net if net is not None else base_label
                return label
    return None
