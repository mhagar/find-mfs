"""
Factoring MS2 evidence to re-rank formula candidates

This module includes subformula assignment + a torch-free executor for MistNet
to generate the re-ranking scores

The reason there's a torch-free executor is to keep this library lightweight
(i.e. so you're not importing PyTorch every time).

Consumes the `.npz` artifact exported by mist-fmfs (the "R&D" repo),
 and calculates log P(formula|MS2), the MS2 term of the stacked
 log-posterior in `find_mfs.scoring`
"""

from .net import MistNetNumpy, CLS_TYPE, FRAG_TYPE
from .featurize import Featurizer, collate, score_candidates
from find_mfs.spectra import build_spectrum, spec_from_pairs
from .assign import (
    assign_spectrum,
    assign_spectrum_batch,
    SpectrumAssignment,
    PeakMatch,
)
from .score import ms2_logits, resolve_ion, ADDUCT_TO_ION
from .tables import ION_LST, ION_TO_ADDUCT
from . import tables

__all__ = [
    # Model
    "MistNetNumpy",
    "CLS_TYPE",
    "FRAG_TYPE",
    # Featurization
    "Featurizer",
    "collate",
    "score_candidates",
    # Spectra
    "build_spectrum",
    "spec_from_pairs",
    # Assignment
    "assign_spectrum",
    "assign_spectrum_batch",
    "SpectrumAssignment",
    "PeakMatch",
    # Scoring
    "ms2_logits",
    "resolve_ion",
    # Vocabularies
    "ION_LST",
    "ION_TO_ADDUCT",
    "ADDUCT_TO_ION",
    "tables",
]
