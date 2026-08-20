"""Tests for the FormulaScorer scoring module."""
import numpy as np
import pytest
from molmass import Formula

from find_mfs import FormulaScorer, FormulaFinder, FormulaSearchResults
from find_mfs.core.finder import FormulaCandidate
from find_mfs.scoring import chem_prior


# A small corpus of common metabolites (halofree)
HALOGEN_FREE_CORPUS = [
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

# A few common halogenated small molecules/drugs, so a corpus can satisfy
# both composition classes (chem_prior.fit() raises if either is too thin).
HALOGENATED_CORPUS = [
    "C6H5Cl",         # chlorobenzene
    "C6H5Br",         # bromobenzene
    "C6H5I",          # iodobenzene
    "CHCl3",          # chloroform
    "C6H4Cl2",        # dichlorobenzene
    "C2H4Br2",        # dibromoethane
    "C6H3Cl3",        # trichlorobenzene
    "C10H7Cl",        # chloronaphthalene
    "C14H9Cl5",       # DDT
    "C8H8Cl2O3",      # 2,4-D
    "C9H10INO3",      # monoiodotyrosine
    "C11H12Cl2N2O5",  # chloramphenicol
]

METABOLITE_CORPUS = HALOGEN_FREE_CORPUS + HALOGENATED_CORPUS


class TestFormulaScorerFit:
    """Test fitting the scorer's chemical prior on a corpus."""

    def test_fit_returns_self(self):
        scorer = FormulaScorer()
        result = scorer.from_corpus(
            corpus_formulae=METABOLITE_CORPUS,
            n_components=3,
        )
        assert result is scorer

    def test_fit_chaining(self):
        scorer = FormulaScorer().from_corpus(
            corpus_formulae=METABOLITE_CORPUS,
            n_components=3,
        )
        # Should be fitted and usable
        score = scorer.log_prior(Formula("C6H12O6"))
        assert isinstance(score, float)

    def test_halogen_free_corpus_raises(self):
        """Fitting requires both classes; a halogen-free corpus raises."""
        with pytest.raises(ValueError, match="Too few halogen examples"):
            FormulaScorer().from_corpus(
                corpus_formulae=HALOGEN_FREE_CORPUS,
                n_components=3,
            )


class TestChemLogPrior:
    """Test the chemical-plausibility log-prior."""

    @pytest.fixture
    def scorer(self):
        return FormulaScorer().from_corpus(
            corpus_formulae=METABOLITE_CORPUS, n_components=3,
        )

    def test_glucose_scores_higher_than_weird(self, scorer):
        """Glucose (normal metabolite) should score higher than a weird formula."""
        glucose_score = scorer.log_prior(Formula("C6H12O6"))
        weird_score = scorer.log_prior(Formula("C2N30H20"))
        assert glucose_score > weird_score

    def test_no_carbon_returns_floor(self, scorer):
        """Formulae without carbon (can't build H/C, O/C ratios) are floored."""
        score = scorer.log_prior(Formula("H2O"))
        assert score == chem_prior._LOG_FLOOR

    def test_scores_are_finite(self, scorer):
        """Gated log-prior scores should be finite (and <= 0)."""
        score = scorer.log_prior(Formula("C6H12O6"))
        assert np.isfinite(score)
        assert score <= 0.0

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
        return FormulaScorer().from_corpus(
            corpus_formulae=METABOLITE_CORPUS, n_components=3,
        )

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


class TestTwoModelGate:
    """Mechanical tests for the per-class models and the plausibility gate."""

    def test_gate_never_positive(self):
        """The gated prior is always <= 0 for carbon-containing formulae."""
        scorer = FormulaScorer().from_corpus(
            corpus_formulae=METABOLITE_CORPUS, n_components=3,
        )
        for f in ("C6H12O6", "C2N30H20", "C10H16N5O13P3", "C8H10N4O2"):
            assert scorer.log_prior(Formula(f)) <= 0.0

    def test_default_loads_both_models(self):
        """The bundled COCONUT prior ships both composition-class models."""
        scorer = FormulaScorer()
        for cls in ("halofree", "halogen"):
            gmm, tau, scale = scorer.chem_prior_params[cls]
            assert gmm is not None
            assert isinstance(tau, float)
            assert isinstance(scale, float)

    def test_default_scores_halogenated_and_not(self):
        """Both a halogenated and a non-halogenated formula score finitely."""
        scorer = FormulaScorer()
        assert np.isfinite(scorer.log_prior(Formula("C6H12O6")))
        assert np.isfinite(scorer.log_prior(Formula("C9H11BrN2O2")))

    def test_save_load_round_trip(self, tmp_path):
        """save() then load() reproduces the per-class models, taus, and scales."""
        scorer = FormulaScorer().from_corpus(
            corpus_formulae=METABOLITE_CORPUS, n_components=3,
        )
        path = tmp_path / "prior.json"
        chem_prior.save(path, scorer.chem_prior_params)

        reloaded = FormulaScorer()
        reloaded.chem_prior_params = chem_prior.load(path)

        for cls in ("halofree", "halogen"):
            _, tau, scale = reloaded.chem_prior_params[cls]
            _, orig_tau, orig_scale = scorer.chem_prior_params[cls]
            assert tau == orig_tau
            assert scale == orig_scale

        # Same formula scores identically after a round trip.
        for f in ("C6H12O6", "C8H10N4O2"):
            assert reloaded.log_prior(Formula(f)) == scorer.log_prior(Formula(f))


class TestSoftGate:
    """The plausibility gate's shape and its live knobs."""

    def test_zero_softness_is_hard_gate(self):
        """chem_softness=0 reproduces the hard hinge: plausible -> exactly 0."""
        scorer = FormulaScorer()
        # A very typical formula sits well above tau -> hard gate gives exactly 0.
        assert scorer.log_prior(Formula("C6H12O6"), softness=0.0) == 0.0

    def test_softness_penalizes_borderline_more_than_typical(self):
        """With a soft ramp, a borderline formula is penalized more than a typical
        one that both clear the hard-gate floor."""
        scorer = FormulaScorer()
        # Both clear the hard gate (would be 0 under the old behavior).
        assert scorer.log_prior(Formula("C6H12O6"), softness=0.0) == 0.0
        assert scorer.log_prior(Formula("C45H64N15O7"), softness=0.0) == 0.0
        # Turning up softness separates them: borderline penalized harder.
        typical = scorer.log_prior(Formula("C6H12O6"), softness=1.0)
        borderline = scorer.log_prior(Formula("C45H64N15O7"), softness=1.0)
        assert borderline < typical < 0.0

    def test_score_accepts_knob_overrides(self):
        """score() honors per-call chem_strength / chem_softness overrides."""
        candidates = [
            FormulaCandidate(
                formula=Formula("C45H64N15O7"),
                error_ppm=1.0, error_da=0.0001, rdbe=10.0,
            ),
        ]
        results = FormulaSearchResults(
            candidates=candidates, query_mass=900.0, query_params={},
        )
        scorer = FormulaScorer()
        scorer.score(results, chem_softness=0.0)
        hard = results[0].chem_logprior
        scorer.score(results, chem_softness=2.0)
        soft = results[0].chem_logprior
        assert hard == 0.0
        assert soft < hard
