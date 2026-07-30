"""Tests for the FormulaScorer scoring module."""
import numpy as np
import pytest
from molmass import Formula

from find_mfs import FormulaScorer, FormulaFinder, FormulaSearchResults
from find_mfs.core.finder import FormulaCandidate


# A small corpus of common metabolites
METABOLITE_CORPUS = [
    "C6H12O6",      # glucose
    "C12H22O11",    # sucrose
    "C27H46O",      # cholesterol
    "C5H9NO4",      # glutamic acid
    "C3H7NO2",      # alanine
    "C6H13NO2",     # leucine
    "C4H8N2O3",     # asparagine
    "C5H11NO2",     # valine
    "C9H11NO3",     # tyrosine
    "C5H9NO2",      # proline
    "C10H16N5O13P3", # ATP
    "C21H28O5",     # cortisol
    "C16H18N2O4S",  # penicillin G
    "C8H10N4O2",    # caffeine
    "C20H25N3O",    # LSD (ergine-related)
]


class TestFormulaScorerFit:
    """Test fitting the scorer's chemical prior on a corpus."""

    def test_fit_returns_self(self):
        scorer = FormulaScorer()
        result = scorer.fit(METABOLITE_CORPUS, n_components=3)
        assert result is scorer

    def test_fit_chaining(self):
        scorer = FormulaScorer().fit(METABOLITE_CORPUS, n_components=3)
        # Should be fitted and usable
        score = scorer.log_prior(Formula("C6H12O6"))
        assert isinstance(score, float)

    def test_log_prior_before_fit_raises_error(self):
        # An unfitted scorer raises an error
        scorer = FormulaScorer()
        with pytest.raises(ValueError):
            scorer.log_prior(Formula("C6H12O6"))


class TestChemLogPrior:
    """Test the chemical-plausibility log-prior."""

    @pytest.fixture
    def scorer(self):
        return FormulaScorer().fit(METABOLITE_CORPUS, n_components=3)

    def test_glucose_scores_higher_than_weird(self, scorer):
        """Glucose (normal metabolite) should score higher than a weird formula."""
        glucose_score = scorer.log_prior(Formula("C6H12O6"))
        weird_score = scorer.log_prior(Formula("C2N30H20"))
        assert glucose_score > weird_score

    def test_no_carbon_returns_zero(self, scorer):
        """Formulae without carbon should get uninformative score (0.0)."""
        score = scorer.log_prior(Formula("H2O"))
        assert score == 0.0

    def test_scores_are_finite(self, scorer):
        """GMM log-density scores should be finite (may be positive)."""
        score = scorer.log_prior(Formula("C6H12O6"))
        assert np.isfinite(score)

    def test_common_metabolite_scores_reasonable(self, scorer):
        """Common metabolites should all get finite scores."""
        for formula_str in METABOLITE_CORPUS:
            score = scorer.log_prior(Formula(formula_str))
            assert isinstance(score, float)
            assert np.isfinite(score)

    def test_works_with_light_formula(self, scorer):
        """Should work with LightFormula via duck typing."""
        from find_mfs.core.light_formula import LightFormula
        lf = LightFormula(
            elements={'C': 6, 'H': 12, 'O': 6},
            charge=0,
            monoisotopic_mass=180.063,
        )
        score = scorer.log_prior(lf)
        assert isinstance(score, float)
        assert np.isfinite(score)


class TestScore:
    """Test the stacked-posterior scoring against FormulaSearchResults."""

    @pytest.fixture
    def scorer(self):
        return FormulaScorer().fit(METABOLITE_CORPUS, n_components=3)

    @pytest.fixture
    def results(self):
        """Create a small FormulaSearchResults for testing (no MS1 envelope)."""
        candidates = [
            FormulaCandidate(
                formula=Formula("C6H12O6"),
                error_ppm=1.0,
                error_da=0.0001,
                rdbe=1.0,
            ),
            FormulaCandidate(
                formula=Formula("C2H8N4O2S2"),
                error_ppm=2.0,
                error_da=0.0002,
                rdbe=1.0,
            ),
        ]
        return FormulaSearchResults(
            candidates=candidates,
            query_mass=180.063,
            query_params={'mass': 180.063},
        )

    def test_score_returns_none(self, scorer, results):
        ret = scorer.score(results, mass_sigma_ppm=2.0)
        assert ret is None

    def test_score_attaches_chem_logprior(self, scorer, results):
        scorer.score(results, mass_sigma_ppm=2.0)
        for candidate in results:
            assert candidate.chem_logprior is not None
            assert isinstance(candidate.chem_logprior, float)

    def test_score_attaches_mass_loglik(self, scorer, results):
        scorer.score(results, mass_sigma_ppm=2.0)
        for candidate in results:
            assert candidate.mass_loglik is not None
            assert isinstance(candidate.mass_loglik, float)

    def test_score_attaches_log_posterior(self, scorer, results):
        scorer.score(results, mass_sigma_ppm=2.0)
        for candidate in results:
            assert candidate.log_posterior is not None
            assert isinstance(candidate.log_posterior, float)

    def test_iso_loglik_none_without_ms1(self, scorer, results):
        """Without an observed envelope, iso_loglik stays None."""
        scorer.score(results, mass_sigma_ppm=2.0)
        for candidate in results:
            assert candidate.iso_loglik is None

    def test_posterior_includes_mass_penalty(self, scorer, results):
        """Posterior should be <= prior due to the (negative) mass penalty."""
        scorer.score(results, mass_sigma_ppm=2.0)
        for candidate in results:
            assert candidate.log_posterior <= candidate.chem_logprior

    def test_glucose_ranked_first_by_posterior(self, scorer, results):
        """Glucose should rank higher than the weird formula by posterior."""
        scorer.score(results, mass_sigma_ppm=2.0)
        sorted_results = results.sort_by_posterior()
        assert sorted_results[0].formula.formula == "C6H12O6"

    def test_sort_by_chem_logprior_method(self, scorer, results):
        """sort_by_chem_logprior sorts ascending when reverse=True."""
        scorer.score(results, mass_sigma_ppm=2.0)
        ascending = results.sort_by_chem_logprior(reverse=True)
        scores = [c.chem_logprior for c in ascending]
        assert scores == sorted(scores)

    def test_sort_by_posterior_method(self, scorer, results):
        """sort_by_posterior sorts descending by default."""
        scorer.score(results, mass_sigma_ppm=2.0)
        descending = results.sort_by_posterior()
        scores = [c.log_posterior for c in descending]
        assert scores == sorted(scores, reverse=True)

    def test_chem_logprior_in_table(self, scorer, results):
        """Chemical prior should appear in table output when present."""
        scorer.score(results, mass_sigma_ppm=2.0)
        table = results.to_table()
        assert "Chem" in table

    def test_score_columns_in_dataframe(self, scorer, results):
        """Score columns should appear in the DataFrame when present."""
        pytest.importorskip("pandas")
        scorer.score(results, mass_sigma_ppm=2.0)
        df = results.to_dataframe()
        assert "chem_logprior" in df.columns
        assert "mass_loglik" in df.columns
        assert "log_posterior" in df.columns

    def test_mass_error_penalizes_score(self, scorer):
        """Larger mass error should give a lower posterior score."""
        low_error = FormulaCandidate(
            formula=Formula("C6H12O6"), error_ppm=0.5, error_da=0.0001, rdbe=1.0,
        )
        high_error = FormulaCandidate(
            formula=Formula("C6H12O6"), error_ppm=4.0, error_da=0.0007, rdbe=1.0,
        )
        results = FormulaSearchResults(
            candidates=[high_error, low_error],
            query_mass=180.063,
            query_params={'mass': 180.063},
        )
        scorer.score(results, mass_sigma_ppm=2.0)
        sorted_results = results.sort_by_posterior()
        assert sorted_results[0].error_ppm == 0.5

    def test_tighter_sigma_increases_mass_penalty(self, scorer):
        """A smaller mass_sigma_ppm should penalize mass error more heavily."""
        def make_results():
            return FormulaSearchResults(
                candidates=[
                    FormulaCandidate(
                        formula=Formula("C6H12O6"), error_ppm=1.0, error_da=0.0001, rdbe=1.0,
                    ),
                    FormulaCandidate(
                        formula=Formula("C6H12O6"), error_ppm=4.0, error_da=0.0007, rdbe=1.0,
                    ),
                ],
                query_mass=180.063,
                query_params={'mass': 180.063},
            )

        loose_results = make_results()
        tight_results = make_results()
        scorer.score(loose_results, mass_sigma_ppm=5.0)
        scorer.score(tight_results, mass_sigma_ppm=1.0)
        # Same formula, different ppm errors — gap should be larger with tight sigma
        loose_gap = abs(loose_results[0].log_posterior - loose_results[1].log_posterior)
        tight_gap = abs(tight_results[0].log_posterior - tight_results[1].log_posterior)
        assert tight_gap > loose_gap

    def test_isotope_scoring_end_to_end(self, scorer):
        """
        With an observed envelope, the matching formula gets a high iso_loglik
        and stays in the results (score-not-omit).
        """
        from find_mfs import get_isotope_envelope

        ion = Formula("C6H13O6+")  # protonated glucose
        env = get_isotope_envelope(ion, mz_tolerance=0.05, threshold=0.001)
        mass = ion.monoisotopic_mass

        finder = FormulaFinder("CHNOPS")
        res = finder.find_formulae(
            mass=mass, charge=1, adduct="H", error_ppm=8.0,
        )
        scorer.score(res, ms1_peaks=env, precursor_mz=mass)

        forms = {c.formula.formula for c in res}
        assert "C6H12O6" in forms  # not omitted

        glucose = next(c for c in res if c.formula.formula == "C6H12O6")
        assert glucose.iso_loglik is not None
        best_iso = max(c.iso_loglik for c in res if c.iso_loglik is not None)
        assert glucose.iso_loglik == best_iso
