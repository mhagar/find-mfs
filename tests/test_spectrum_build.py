"""Tests for the canonical spectrum constructors and the intensity convention.

`normalize` used to default to a 0-100 scale while the MS2 stage built its own
base-peak-1.0 spectra. Those are now unified on base-peak-1.0.

The convention is worth pinning hard: intensity is fed straight into the trained
MS2 reranker as a feature, so getting the scale wrong does not raise -- it just
silently degrades scores.
"""

import numpy as np
import pytest

from find_mfs.spectra import build_spectrum, normalize, spec_from_pairs, to_spec_arr


def test_base_peak_is_one():
    """THE convention. If this changes, the MS2 reranker's inputs change."""
    spec = build_spectrum([100.0, 200.0, 300.0], [5.0, 20.0, 10.0])
    assert spec["intsy"].max() == 1.0
    np.testing.assert_allclose(spec["intsy"], [0.25, 1.0, 0.5])


def test_normalize_default_is_base_peak_one():
    spec = normalize(to_spec_arr([1.0, 2.0], [5.0, 10.0]))
    assert spec["intsy"].max() == 1.0
    np.testing.assert_allclose(spec["intsy"], [0.5, 1.0])


def test_normalize_custom_base():
    spec = normalize(to_spec_arr([1.0, 2.0], [5.0, 10.0]), base=100.0)
    np.testing.assert_allclose(spec["intsy"], [50.0, 100.0])


def test_normalize_does_not_mutate_input():
    original = to_spec_arr([1.0, 2.0], [5.0, 10.0])
    normalize(original)
    np.testing.assert_allclose(original["intsy"], [5.0, 10.0])


def test_sorted_by_mz():
    spec = build_spectrum([300.0, 100.0, 200.0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(spec["mz"], [100.0, 200.0, 300.0])


def test_intensities_follow_their_mz_when_sorted():
    """Sorting must permute both fields together."""
    spec = build_spectrum([300.0, 100.0, 200.0], [3.0, 1.0, 2.0])
    np.testing.assert_allclose(spec["mz"], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(spec["intsy"], [1 / 3, 2 / 3, 1.0])


def test_drop_zero():
    assert len(build_spectrum([1.0, 2.0, 3.0], [0.0, 3.0, -1.0])) == 1
    assert len(build_spectrum([1.0, 2.0, 3.0], [0.0, 3.0, -1.0], drop_zero=False)) == 3


def test_normalize_intensity_off_preserves_raw_scale():
    spec = build_spectrum([1.0, 2.0], [5.0, 10.0], normalize_intensity=False)
    np.testing.assert_allclose(spec["intsy"], [5.0, 10.0])


def test_empty_and_degenerate_inputs():
    assert len(build_spectrum([], [])) == 0
    # all-zero intensities: drop_zero removes everything rather than dividing by 0
    assert len(build_spectrum([1.0, 2.0], [0.0, 0.0])) == 0
    # ...and with drop_zero off, normalize must not produce NaN
    spec = build_spectrum([1.0, 2.0], [0.0, 0.0], drop_zero=False)
    assert not np.isnan(spec["intsy"]).any()


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        build_spectrum([1.0, 2.0], [1.0])


def test_spec_from_pairs_matches_build_spectrum():
    pairs = np.array([[200.0, 5.0], [100.0, 20.0]])
    np.testing.assert_array_equal(
        spec_from_pairs(pairs), build_spectrum(pairs[:, 0], pairs[:, 1])
    )


@pytest.mark.parametrize("bad", [np.zeros(3), np.zeros((2, 3)), np.zeros((2, 2, 2))])
def test_spec_from_pairs_rejects_wrong_shape(bad):
    with pytest.raises(ValueError, match=r"expected \(N, 2\)"):
        spec_from_pairs(bad)
