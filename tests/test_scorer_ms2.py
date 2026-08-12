"""
Tests for the MS2 term wired into FormulaScorer.

`score()` populates the per-candidate `ms2_logit` only; the set-normalized term
is applied at ranking time (see tests/test_ms2_posterior.py). These tests cover
the wiring: attachment, the top-N cascade, and the ways the term can be absent.
"""

from pathlib import Path

import numpy as np
import pytest

from find_mfs import FormulaScorer, get_finder

ARTIFACT = Path(
    "/home/mostafa/projects/mist-fmfs/artifacts/mistnet_shipped.npz"
)

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="needs the exported MistNet .npz artifact"
)

PRECURSOR = 515.3228
PEAKS = np.array([
    [70.0651, 0.30], [86.0964, 0.55], [110.0713, 0.22], [136.0757, 1.00],
    [166.0862, 0.18], [249.1234, 0.14], [498.3068, 0.42], [515.3228, 0.60],
])


def _results(adduct="H", charge=1, error_ppm=2.0):
    return get_finder("CHNOPS").find_formulae(
        mass=PRECURSOR, charge=charge, adduct=adduct, error_ppm=error_ppm
    )


@pytest.fixture(scope="module")
def scorer():
    return FormulaScorer.default().with_ms2(ARTIFACT)


def test_with_ms2_accepts_a_path_and_a_model(scorer):
    from find_mfs.ms2 import MistNetNumpy

    assert scorer.has_ms2
    assert not FormulaScorer.default().has_ms2
    # an already-loaded model is accepted unchanged
    model = MistNetNumpy.from_npz(ARTIFACT)
    assert FormulaScorer.default().with_ms2(model)._ms2_model is model


def test_ms2_peaks_without_a_model_is_an_error():
    res = _results()
    with pytest.raises(ValueError, match="no MS2 reranker is attached"):
        FormulaScorer.default().score(
            res, precursor_mz=PRECURSOR, ms2_peaks=PEAKS
        )


def test_score_populates_ms2_logit(scorer):
    res = _results()
    scorer.score(res, precursor_mz=PRECURSOR, ms2_peaks=PEAKS,
                 ms2_use_halogens=False, ms2_top_n=None)
    logits = [c.ms2_logit for c in res.candidates]
    assert all(v is not None for v in logits)
    assert len(set(logits)) > 1, "every candidate got the same logit"


def test_no_ms2_peaks_leaves_the_term_absent(scorer):
    res = _results()
    scorer.score(res, precursor_mz=PRECURSOR)
    assert all(c.ms2_logit is None for c in res.candidates)
    # ...and the posterior is then just the per-candidate terms
    np.testing.assert_allclose(
        res.log_posterior(), [c.log_posterior for c in res.candidates]
    )


def test_top_n_limits_the_cascade(scorer):
    res = _results(error_ppm=10.0)
    n = len(res)
    assert n > 20, "need a decent candidate pool to exercise the cut"

    scorer.score(res, precursor_mz=PRECURSOR, ms2_peaks=PEAKS,
                 ms2_use_halogens=False, ms2_top_n=10)
    scored = [c for c in res.candidates if c.ms2_logit is not None]
    assert len(scored) == 10

    # the cut must be taken on the cheap terms, best-first
    best_base = sorted(
        res.candidates, key=lambda c: c.log_posterior, reverse=True
    )[:10]
    assert {id(c) for c in scored} == {id(c) for c in best_base}


def test_unscored_candidates_never_outrank_scored_ones(scorer):
    """The cascade must not reward a candidate for being skipped."""
    res = _results(error_ppm=10.0)
    scorer.score(res, precursor_mz=PRECURSOR, ms2_peaks=PEAKS,
                 ms2_use_halogens=False, ms2_top_n=10)
    ll = res.ms2_loglik()
    unscored = [v for v, c in zip(ll, res.candidates) if c.ms2_logit is None]
    scored = [v for v, c in zip(ll, res.candidates) if c.ms2_logit is not None]
    assert max(unscored) <= min(scored)


def test_ion_outside_the_reranker_vocabulary_is_skipped(scorer):
    """Negative mode has no column in the 7-ion head -- skip, don't crash."""
    res = get_finder("CHNOPS").find_formulae(
        mass=PRECURSOR, charge=-1, adduct="-H", error_ppm=2.0
    )
    scorer.score(res, precursor_mz=PRECURSOR, ms2_peaks=PEAKS,
                 ms2_use_halogens=False)
    assert all(c.ms2_logit is None for c in res.candidates)
    np.testing.assert_array_equal(res.ms2_loglik(), np.zeros(len(res)))


def test_ms2_changes_the_ranking(scorer):
    res = _results(error_ppm=10.0)
    scorer.score(res, precursor_mz=PRECURSOR, ms2_peaks=PEAKS,
                 ms2_use_halogens=False, ms2_top_n=None)
    with_ms2 = [c.formula.formula for c in
                res.sort_by_posterior(ms2_weight=1.0).candidates[:10]]
    without = [c.formula.formula for c in
               res.sort_by_posterior(ms2_weight=0.0).candidates[:10]]
    assert with_ms2 != without


def test_score_is_idempotent(scorer):
    """Re-scoring the same results must not accumulate or drift."""
    res = _results()
    scorer.score(res, precursor_mz=PRECURSOR, ms2_peaks=PEAKS,
                 ms2_use_halogens=False, ms2_top_n=None)
    first = [(c.log_posterior, c.ms2_logit) for c in res.candidates]
    scorer.score(res, precursor_mz=PRECURSOR, ms2_peaks=PEAKS,
                 ms2_use_halogens=False, ms2_top_n=None)
    second = [(c.log_posterior, c.ms2_logit) for c in res.candidates]
    assert first == second
