"""
Turn MS2 subformula assignments into MistNet's input tensors (numpy only)

This is a port of `mist_cf.assignment.featurize`.

This file defines the feature contract:
mist-fmfs imports it for training too, rather than keeping a parallel torch copy that can silently drift
 out of sync with this one (see `docs/ms2_port_plan.md` Sec 1.1)

Per candidate, row 0 is the CLS token (the precursor/root formula) and rows 1..N
are the matched subpeak subformulae, ordered most-intense-first.
"""

from __future__ import annotations

import numpy as np

from .net import CLS_TYPE, FRAG_TYPE
from .tables import (
    MAX_INSTR_IDX,
    NUM_ELEMENTS,
    NUM_ION,
    NUM_ION_CLASSES,
    clipped_ppm_single_norm,
    formula_to_dense,
    get_cls_mass_diff,
    get_instr_idx,
    get_ion_idx,
    norm_mass_diff_ppm,
)


class Featurizer:
    """
    Builds MistNet feature dicts from subformula assignments

    Args:
        max_subpeak: keep at most this many subpeaks (most intense first).
        ablate_cls_error: zero the CLS precursor-mass-error feature. Must be the
            negation of the checkpoint's `cls_mass_diff`.
        collapse_ions: use the narrowed twin-blind ion one-hot. Must match the
            checkpoint's `collapse_ions`.

    Prefer :meth:`from_model`, which reads all three off a loaded model so they
    cannot be set inconsistently.
    """

    def __init__(
            self,
            max_subpeak: int,
            ablate_cls_error: bool = True,
            collapse_ions: bool = False,
    ):
        self.max_subpeak = max_subpeak
        self.ablate_cls_error = ablate_cls_error
        self.collapse_ions = collapse_ions
        self.ion_mat = np.eye(NUM_ION_CLASSES if collapse_ions else NUM_ION)
        self.instrument_mat = np.eye(MAX_INSTR_IDX)

    @classmethod
    def from_model(cls, model) -> "Featurizer":
        """Configure from a :class:`~find_mfs.ms2.net.MistNetNumpy`."""
        return cls(
            max_subpeak=model.max_subpeak,
            ablate_cls_error=not model.cls_mass_diff,
            collapse_ions=model.collapse_ions,
        )

    def get_ion_embed(self, ion: str) -> np.ndarray:
        return self.ion_mat[get_ion_idx(ion, collapse=self.collapse_ions)]

    def get_instrument_embed(self, instrument: str) -> np.ndarray:
        return self.instrument_mat[get_instr_idx(instrument)]

    def candidate(
            self,
            assignment,
            *,
            parentmass,
            instrument,
            name=None,
    ) -> dict:
        """
        Feature dict for one ``(root formula, ion)`` candidate

        `assignment` needs `.root_formula`, `.ion` and `.matches`,
        where each match has `.formula`, `.intensity` and `.ppm` -- i.e. a `SpectrumAssignment`.
        """
        form, ion = assignment.root_formula, assignment.ion

        embed_form = formula_to_dense(form)
        embed_ion = self.get_ion_embed(ion)

        if self.ablate_cls_error:
            cls_ppm = 0.0
        else:
            cls_ppm = clipped_ppm_single_norm(
                get_cls_mass_diff(parentmass, form=form, ion=ion, corr_electrons=True),
                parentmass,
            )

        # Keep the most intense subpeaks when truncating. The transformer is
        # permutation invariant, so only *which* subpeaks survive matters.
        matches = sorted(assignment.matches, key=lambda m: m.intensity, reverse=True)
        matches = matches[: self.max_subpeak]

        form_vecs = [embed_form] + [formula_to_dense(m.formula) for m in matches]
        ion_vecs = [embed_ion] + [embed_ion] * len(matches)
        peak_types = [CLS_TYPE] + [FRAG_TYPE] * len(matches)
        frag_intens = [1.0] + [float(m.intensity) for m in matches]

        if matches:
            rel = np.array([m.ppm for m in matches], dtype=np.float64)
            rel_mass_diffs = [cls_ppm] + norm_mass_diff_ppm(rel).tolist()
        else:
            rel_mass_diffs = [cls_ppm]

        instrument_vec = self.get_instrument_embed(instrument)

        return {
            "name": name,
            "formula": form,
            "ion": ion,
            "parentmass": float(parentmass),
            "instrument": instrument,
            "form_vecs": np.asarray(form_vecs, dtype=np.float64),
            "ion_vecs": np.asarray(ion_vecs, dtype=np.float64),
            "instrument_vecs": np.asarray(
                [instrument_vec] * len(form_vecs), dtype=np.float64
            ),
            "peak_types": peak_types,
            "frag_intens": frag_intens,
            "rel_mass_diffs": rel_mass_diffs,
        }


def collate(
        cands: list[dict],
        pad_to: int | None = None,
) -> dict:
    """
    Stack candidate feature dicts into the arrays `MistNetNumpy.forward` takes

    Args:
        cands: feature dicts from :meth:`Featurizer.candidate`.
        pad_to: pad every candidate to exactly this many rows. **Pass
            ``max_subpeak + 1``.** Padded positions are not masked out of
            attention (a reproduced upstream bug -- see :mod:`find_mfs.ms2.net`),
            so scores depend on the padded width; a fixed width makes output
            deterministic and independent of how candidates were batched.
            ``None`` pads to the batch maximum, reproducing the reference's
            batch-dependent behaviour, and is intended only for parity testing.
    """
    lens = np.array([len(c["peak_types"]) for c in cands])
    width = int(lens.max()) if pad_to is None else int(pad_to)
    if width < lens.max():
        raise ValueError(f"pad_to={width} is shorter than the longest candidate ({lens.max()})")

    n = len(cands)
    out = {
        "types": np.zeros((n, width), dtype=np.int64),
        "form_vec": np.zeros((n, width, NUM_ELEMENTS), dtype=np.float64),
        "ion_vec": np.zeros((n, width, cands[0]["ion_vecs"].shape[1]), dtype=np.float64),
        "instrument_vec": np.zeros((n, width, MAX_INSTR_IDX), dtype=np.float64),
        "intens": np.zeros((n, width), dtype=np.float64),
        "rel_mass_diffs": np.zeros((n, width), dtype=np.float64),
    }
    for i, c in enumerate(cands):
        k = len(c["peak_types"])
        out["types"][i, :k] = c["peak_types"]
        out["form_vec"][i, :k] = c["form_vecs"]
        out["ion_vec"][i, :k] = c["ion_vecs"]
        out["instrument_vec"][i, :k] = c["instrument_vecs"]
        out["intens"][i, :k] = c["frag_intens"]
        out["rel_mass_diffs"][i, :k] = c["rel_mass_diffs"]

    out["num_peaks"] = lens.astype(np.int64)
    # carry-through
    out["names"] = [c["name"] for c in cands]
    out["str_forms"] = [c["formula"] for c in cands]
    out["str_ions"] = [c["ion"] for c in cands]
    out["parentmasses"] = [c["parentmass"] for c in cands]
    return out


def score_candidates(
        model,
        cands: list[dict],
        batch_size: int = 256,
) -> np.ndarray:
    """
    Run `model` over candidate feature dicts, returning `(len(cands),)` logits.

    Always pads to `model.max_subpeak + 1` so a candidate's score does not
    depend on which batch it landed in.
    """
    pad_to = model.max_subpeak + 1
    scores = np.empty(len(cands), dtype=np.float64)
    for start in range(0, len(cands), batch_size):
        chunk = cands[start : start + batch_size]
        b = collate(chunk, pad_to=pad_to)
        scores[start : start + len(chunk)] = model.forward(
            b["num_peaks"], b["types"], b["form_vec"], b["ion_vec"],
            b["instrument_vec"], b["intens"], b["rel_mass_diffs"],
        )
    return scores
