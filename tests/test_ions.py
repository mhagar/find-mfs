"""
Tests for the ion-type vocabulary and the deconvolution mass model.

The model is M = (D - offset)/n with D = z*mz + z*m_e. These tests pin the three
independent axes (charge in D, adduct in offset, multimer in n) by round-tripping
a known analyte mass through synthetic ion m/z values.
"""
import numpy as np

from find_mfs.spectra.ions import (
    ION_VOCAB, IonType, deconv_mass, ELECTRON_MASS, _H, _NA,
)

VOCAB = {ion.label: ion for ion in ION_VOCAB}


def test_vocab_is_ion_types_and_labels_unique():
    assert all(isinstance(i, IonType) for i in ION_VOCAB)
    labels = [i.label for i in ION_VOCAB]
    assert len(labels) == len(set(labels))


def test_mh_recovers_M_with_electron_mass():
    M = 300.0
    mz = M + _H - ELECTRON_MASS          # [M+H]+ observed m/z
    D = deconv_mass(mz, 1)
    assert D == 300.0 + _H               # electron folded back exactly
    assert VOCAB["[M+H]+"].neutral_mass(D) == 300.0


def test_multiply_charged_round_trip():
    """[M+2H]2+: charge lives in D, offset is 2*m_H, n=1."""
    M = 512.3
    mz = (M + 2 * _H - 2 * ELECTRON_MASS) / 2
    D = deconv_mass(mz, 2)
    assert VOCAB["[M+2H]2+"].neutral_mass(D) == np.float64(M) or \
        abs(VOCAB["[M+2H]2+"].neutral_mass(D) - M) < 1e-9


def test_multimer_round_trip():
    """[2M+H]+: multimer lives in n=2."""
    M = 250.1
    mz = 2 * M + _H - ELECTRON_MASS
    D = deconv_mass(mz, 1)
    assert abs(VOCAB["[2M+H]+"].neutral_mass(D) - M) < 1e-9
    # Read as a monomer [M+H]+ it would (wrongly) imply ~2M -- the two disagree.
    assert VOCAB["[M+H]+"].neutral_mass(D) > 1.5 * M


def test_offsets_have_expected_signs():
    assert VOCAB["[M+H]+"].offset > 0
    assert VOCAB["[M+Na]+"].offset > VOCAB["[M+H]+"].offset
    assert VOCAB["[M+H-H2O]+"].offset < 0        # net loss
    assert VOCAB["[M]+"].offset == 0.0


def test_deconv_mass_vectorized():
    mz = np.array([100.0, 200.0])
    D = deconv_mass(mz, 1)
    assert np.allclose(D, mz + ELECTRON_MASS)
