"""
Tests for the logit/loglik split on the MS2 term.

The MS2 reranker's normalized output is a softmax over the candidate set, so it
is only meaningful relative to the set it was computed over. Storing it on a
candidate meant it silently went stale as soon as the set was filtered
(measured: a normalized set summing to 1.0 summed to 0.26 after filtering).

The invariant these tests pin: **everything stored on a candidate is
per-candidate and stable; everything set-dependent is computed on demand.**
"""

import numpy as np
import pytest

from find_mfs import get_finder
from find_mfs.core.results import FormulaSearchResults


@pytest.fixture(scope="module")
def scored():
    """A results set with per-candidate terms and MS2 logits populated."""
    res = get_finder("CHNOPS").find_formulae(
        mass=180.0634, charge=1, adduct="H", error_ppm=50.0
    )
    cands = res.candidates
    rng = np.random.default_rng(0)
    for i, c in enumerate(cands):
        c.ms2_logit = float(rng.normal(0, 2))
        c.chem_logprior = -float(i) * 0.01
        c.mass_loglik = -abs(c.error_ppm) / 10
        c.log_posterior = c.chem_logprior + c.mass_loglik
    assert len(cands) > 10
    return FormulaSearchResults(
        candidates=cands, query_mass=res.query_mass, query_params=res.query_params
    )


def test_loglik_normalizes_over_the_current_set(scored):
    assert np.exp(scored.ms2_loglik()).sum() == pytest.approx(1.0)


def test_loglik_renormalizes_after_filtering(scored):
    """The bug this split exists to fix: a filtered set must still sum to 1."""
    subset = FormulaSearchResults(
        candidates=scored.candidates[:5],
        query_mass=scored.query_mass,
        query_params=scored.query_params,
    )
    assert np.exp(subset.ms2_loglik()).sum() == pytest.approx(1.0)


def test_logits_survive_filtering_unchanged(scored):
    """The per-candidate half must be stable -- that is why it is stored."""
    before = [c.ms2_logit for c in scored.candidates[:5]]
    subset = FormulaSearchResults(
        candidates=scored.candidates[:5],
        query_mass=scored.query_mass,
        query_params=scored.query_params,
    )
    assert [c.ms2_logit for c in subset.candidates] == before


def test_sorting_does_not_change_the_normalization(scored):
    """Sorting keeps the set intact, so the term must be unaffected."""
    assert np.exp(scored.sort_by_posterior().ms2_loglik()).sum() == pytest.approx(1.0)


def test_temperature_sharpens_and_flattens(scored):
    """Lower T concentrates probability mass; higher T spreads it."""
    def spread(t):
        return np.exp(scored.ms2_loglik(temperature=t)).max()

    assert spread(0.5) > spread(1.0) > spread(4.0)


def test_invalid_temperature_rejected(scored):
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            scored.ms2_loglik(temperature=bad)


def test_unscored_candidates_are_floored_not_favoured(scored):
    """
    An unscored candidate must never outrank a scored one on MS2 evidence.

    Giving unscored candidates 0.0 would do exactly that, since every scored
    log-probability is negative.
    """
    cands = list(scored.candidates)
    cands[0].ms2_logit = None
    res = FormulaSearchResults(
        candidates=cands, query_mass=scored.query_mass, query_params=scored.query_params
    )
    try:
        ll = res.ms2_loglik()
        assert ll[0] == ll[1:].min()
        assert ll[0] <= 0.0

        nan_policy = res.ms2_loglik(unscored="nan")
        assert np.isnan(nan_policy[0])

        with pytest.raises(ValueError, match="unscored must be"):
            res.ms2_loglik(unscored="zero")
    finally:
        cands[0].ms2_logit = float(np.random.default_rng(1).normal())


def test_no_ms2_evidence_yields_a_zero_term(scored):
    """With nothing scored the term is absent, not -inf."""
    cands = [
        type(c)(formula=c.formula, error_ppm=c.error_ppm, error_da=c.error_da,
                rdbe=c.rdbe, log_posterior=-1.0)
        for c in scored.candidates
    ]
    res = FormulaSearchResults(candidates=cands, query_mass=scored.query_mass)
    np.testing.assert_array_equal(res.ms2_loglik(), np.zeros(len(cands)))
    np.testing.assert_allclose(res.log_posterior(), np.full(len(cands), -1.0))


def test_weight_zero_reproduces_per_candidate_posterior(scored):
    np.testing.assert_allclose(
        scored.log_posterior(ms2_weight=0.0),
        [c.log_posterior for c in scored.candidates],
    )


def test_weight_and_temperature_are_degenerate_for_ranking(scored):
    """Only w/T affects ordering -- the documented tradeoff."""
    a = scored.sort_by_posterior(ms2_weight=1.0, ms2_temperature=0.5)
    b = scored.sort_by_posterior(ms2_weight=2.0, ms2_temperature=1.0)
    assert [c.formula.formula for c in a.candidates] == \
           [c.formula.formula for c in b.candidates]


def test_sort_by_posterior_uses_the_ms2_term(scored):
    """Ordering must actually respond to MS2, not silently ignore it."""
    with_ms2 = scored.sort_by_posterior(ms2_weight=5.0)
    without = scored.sort_by_posterior(ms2_weight=0.0)
    assert [c.formula.formula for c in with_ms2.candidates] != \
           [c.formula.formula for c in without.candidates]
    # and the top hit under a heavy MS2 weight should be the best logit
    best = max(scored.candidates, key=lambda c: c.ms2_logit)
    assert with_ms2.candidates[0].formula.formula == best.formula.formula
