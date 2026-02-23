"""Tests for the FormulaPrior scoring module."""
import pytest
from molmass import Formula

from find_mfs import FormulaPrior, FormulaFinder, FormulaSearchResults
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


class TestFormulaPriorFit:
    """Test fitting the prior on a corpus."""

    def test_fit_returns_self(self):
        prior = FormulaPrior()
        result = prior.fit(METABOLITE_CORPUS)
        assert result is prior

    def test_fit_chaining(self):
        prior = FormulaPrior().fit(METABOLITE_CORPUS)
        # Should be fitted and usable
        score = prior.log_prior(Formula("C6H12O6"))
        assert isinstance(score, float)

    def test_log_prior_before_fit_raises(self):
        prior = FormulaPrior()
        with pytest.raises(RuntimeError, match="fit"):
            prior.log_prior(Formula("C6H12O6"))


class TestFormulaPriorScoring:
    """Test scoring individual formulae."""

    @pytest.fixture
    def prior(self):
        return FormulaPrior().fit(METABOLITE_CORPUS)

    def test_glucose_scores_higher_than_weird(self, prior):
        """Glucose (normal metabolite) should score higher than a weird formula."""
        glucose_score = prior.log_prior(Formula("C6H12O6"))
        weird_score = prior.log_prior(Formula("C2N30H20"))
        assert glucose_score > weird_score

    def test_no_carbon_returns_zero(self, prior):
        """Formulae without carbon should get uninformative score (0.0)."""
        score = prior.log_prior(Formula("H2O"))
        assert score == 0.0

    def test_scores_are_negative(self, prior):
        """Log probabilities should be negative (or zero for no-carbon)."""
        score = prior.log_prior(Formula("C6H12O6"))
        assert score < 0.0

    def test_common_metabolite_scores_reasonable(self, prior):
        """Common metabolites should all get finite scores."""
        for formula_str in METABOLITE_CORPUS:
            score = prior.log_prior(Formula(formula_str))
            assert isinstance(score, float)
            assert score != float('-inf')
            assert score != float('inf')

    def test_works_with_light_formula(self, prior):
        """Should work with LightFormula via duck typing."""
        from find_mfs.core.light_formula import LightFormula
        lf = LightFormula(
            elements={'C': 6, 'H': 12, 'O': 6},
            charge=0,
            monoisotopic_mass=180.063,
        )
        score = prior.log_prior(lf)
        assert isinstance(score, float)
        assert score < 0.0


class TestScoreResults:
    """Test integration with FormulaSearchResults."""

    @pytest.fixture
    def prior(self):
        return FormulaPrior().fit(METABOLITE_CORPUS)

    @pytest.fixture
    def results(self):
        """Create a small FormulaSearchResults for testing."""
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

    def test_score_results_returns_none(self, prior, results):
        ret = prior.score_results(results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
        assert ret is None

    def test_score_results_attaches_prior_scores(self, prior, results):
        prior.score_results(results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
        for candidate in results:
            assert candidate.prior_score is not None
            assert isinstance(candidate.prior_score, float)

    def test_score_results_attaches_posterior_scores(self, prior, results):
        prior.score_results(results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
        for candidate in results:
            assert candidate.posterior_score is not None
            assert isinstance(candidate.posterior_score, float)

    def test_posterior_includes_mass_penalty(self, prior, results):
        """Posterior should be lower than prior due to mass error penalty."""
        prior.score_results(results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
        for candidate in results:
            assert candidate.posterior_score <= candidate.prior_score

    def test_glucose_ranked_first_by_posterior(self, prior, results):
        """Glucose should rank higher than the weird formula by posterior."""
        prior.score_results(results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
        sorted_results = results.sort_by_posterior()
        assert sorted_results[0].formula.formula == "C6H12O6"

    def test_sort_by_prior_method(self, prior, results):
        """Test the sort_by_prior method on FormulaSearchResults."""
        prior.score_results(results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
        ascending = results.sort_by_prior(reverse=True)
        scores = [c.prior_score for c in ascending]
        assert scores == sorted(scores)

    def test_sort_by_posterior_method(self, prior, results):
        """Test the sort_by_posterior method on FormulaSearchResults."""
        prior.score_results(results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
        descending = results.sort_by_posterior()
        scores = [c.posterior_score for c in descending]
        assert scores == sorted(scores, reverse=True)

    def test_prior_score_in_table(self, prior, results):
        """Prior score should appear in table output when present."""
        prior.score_results(results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
        table = results.to_table()
        assert "Prior" in table

    def test_prior_score_in_dataframe(self, prior, results):
        """Prior score should appear in DataFrame when present."""
        pd = pytest.importorskip("pandas")
        prior.score_results(results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
        df = results.to_dataframe()
        assert "prior_score" in df.columns

    def test_mass_error_penalizes_score(self, prior):
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
        prior.score_results(results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
        sorted_results = results.sort_by_posterior()
        assert sorted_results[0].error_ppm == 0.5

    def test_tighter_sigma_increases_mass_penalty(self, prior):
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
        prior.score_results(loose_results, mass_sigma_ppm=5.0, isotope_sigma=0.05)
        prior.score_results(tight_results, mass_sigma_ppm=1.0, isotope_sigma=0.05)
        # Same formula, different ppm errors — gap should be larger with tight sigma
        loose_gap = abs(loose_results[0].posterior_score - loose_results[1].posterior_score)
        tight_gap = abs(tight_results[0].posterior_score - tight_results[1].posterior_score)
        assert tight_gap > loose_gap
