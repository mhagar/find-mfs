"""
Mass-difference network (MDN): resolve the adduct identity of a base envelope.

The core is candidate-M clustering. Every signal group, read as every ion type in
the vocabulary, implies a neutral analyte mass ``M = (D - offset)/n``. Groups that
belong to the same analyte agree on one M; the job is to find, among the M values
the *base* group could take, the one the most (and highest-prior) other groups also
support. That single step resolves degenerate ions -- an NH3 gap that could be an
adduct or a loss is decided by which interpretation shares an M with the rest of
the group -- and it is not metal-dependent: metals just carry a high prior, so they
anchor strongly when present, but a metal-free cluster (e.g. [M+H]+, [M+H-H2O]+,
[M+NH4]+) still pins one M.

See `ions.py` for the mass model. Explicit mass-difference *edges* are an optional
prefilter/visualisation aid, not needed here: clustering candidate M's is the same
relation expressed directly, and it also spans charge states and multimers, which a
fixed pairwise delta table cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grouping import GroupedSpectrum
from .ions import IonType, ION_VOCAB


@dataclass(slots=True)
class Solution:
    """Result of resolving one base group's adduct via the MDN."""
    base_gid: int
    M: float                          # resolved analyte neutral mass
    base_ion: IonType                 # winning ion type for the base group
    near_tie_ions: list[IonType]      # base ion types within the tie margin (winner first)
    score: float
    assignments: dict[int, IonType] = field(default_factory=dict)  # gid -> ion type

    def candidate_adducts(self) -> list[tuple[str | None, int]]:
        """
        `(adduct, charge)` pairs for the base group, best first, shaped for
        `normalize_adducts`. Deduplicated on (adduct, charge).
        """
        out: list[tuple[str | None, int]] = []
        for ion in self.near_tie_ions:
            pair = (ion.adduct_str, ion.charge)
            if pair not in out:
                out.append(pair)
        return out


def solve_for_base(
    grouped: GroupedSpectrum,
    base_gid: int,
    vocab: list[IonType] = ION_VOCAB,
    *,
    tol_da: float = 0.005,
    tol_ppm: float = 5.0,
    tie_margin: float = 0.05,
    intensity_weight: float = 1.0,
    write_back: bool = True,
) -> Solution:
    """
    Resolve the base group's adduct/charge and the analyte's neutral mass M.

    For each ion-type hypothesis the base group could be, score the M it implies by
    how many other groups support that same M, each weighted by its ion-type prior
    AND its relative intensity. The best-scoring hypothesis wins; near-ties within
    `tie_margin` (relative) are returned too, for MS2 to break downstream.

    The intensity weighting keeps the network from fixating on a coincidental pair
    of trace peaks over the dominant species: `intensity_weight` is the exponent on
    each group's base-peak-relative intensity (1.0 = linear; 0.0 recovers the old
    unweighted behaviour; 0.5 is a gentler middle ground).

    When `write_back`, fills `adduct_label`, `M`, and `component_id` on `grouped`
    for every group the winning M explains (single component sharing `base_gid`).
    """
    D = grouped.deconv_mass                      # (n_groups,)
    group_charge = grouped.charge                # (n_groups,)
    n_groups = grouped.n_groups
    priors = np.array([ion.prior for ion in vocab], dtype=np.float64)

    # cand_M[g, k] = neutral mass if group g is ion type k. Only ion types whose
    # charge matches the group's inferred charge are considered -- the isotope
    # spacing already fixed the charge, so a z=1 envelope must not be reinterpreted
    # as a z=2 ion (and vice versa). Mismatches stay NaN and never match.
    cand_M = np.full((n_groups, len(vocab)), np.nan, dtype=np.float64)
    for k, ion in enumerate(vocab):
        rows = group_charge == ion.charge
        cand_M[rows, k] = (D[rows] - ion.offset) / ion.n

    # Per-group intensity weight: base-peak-relative apex intensity, raised to
    # `intensity_weight`. Trace peaks contribute ~0, so a low-abundance network
    # cannot outvote the dominant species.
    apex = grouped.apex_intensity.astype(np.float64)
    apex_max = apex.max() if len(apex) and apex.max() > 0 else 1.0
    weight = (apex / apex_max) ** intensity_weight

    def support_for(m_base: float) -> tuple[float, dict[int, int]]:
        """Total support score for hypothesis M=m_base, and each group's best ion k."""
        tol = max(tol_da, m_base * tol_ppm * 1e-6)
        assign: dict[int, int] = {}
        score = 0.0
        for g in range(n_groups):
            matches = np.abs(cand_M[g] - m_base) <= tol
            if not np.any(matches):
                continue
            # Best-prior ion type for this group at this M.
            k = int(np.argmax(np.where(matches, priors, -np.inf)))
            assign[g] = k
            score += priors[k] * weight[g]
        return score, assign

    # Score every hypothesis the base group could take (charge-consistent ones).
    hyps = []  # (score, k_base, m_base, assign)
    for k_base in range(len(vocab)):
        m_base = cand_M[base_gid, k_base]
        if np.isnan(m_base):
            continue
        score, assign = support_for(m_base)
        hyps.append((score, k_base, m_base, assign))

    # Prefer higher score, then break ties toward simpler ions (lower multiplicity,
    # then lower charge) so equivalent readings don't surface as odd labels.
    hyps.sort(key=lambda h: (h[0], -vocab[h[1]].n, -vocab[h[1]].charge), reverse=True)
    best_score, best_k, best_M, best_assign = hyps[0]

    # Near-ties: distinct-M hypotheses whose score is within the margin.
    winner_ion = vocab[best_k]
    near_tie_ions: list[IonType] = [winner_ion]
    tol_M = max(tol_da, best_M * tol_ppm * 1e-6)
    for score, k, m, _ in hyps[1:]:
        if score >= best_score * (1.0 - tie_margin) and abs(m - best_M) > tol_M:
            ion = vocab[k]
            if ion not in near_tie_ions:
                near_tie_ions.append(ion)

    assignments = {g: vocab[k] for g, k in best_assign.items()}

    if write_back:
        for g, ion in assignments.items():
            grouped.adduct_label[g] = ion.label
            grouped.M[g] = best_M
            grouped.component_id[g] = base_gid

    return Solution(
        base_gid=base_gid,
        M=float(best_M),
        base_ion=winner_ion,
        near_tie_ions=near_tie_ions,
        score=float(best_score),
        assignments=assignments,
    )
