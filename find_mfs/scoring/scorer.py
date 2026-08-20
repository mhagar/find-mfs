"""
FormulaScorer: a stacked-likelihood scorer for molecular formula candidates.

The scorer holds a corpus-derived chemical prior over composition features and,
given an observed MS1 peak list, folds in isotope and precursor-mass likelihoods
to produce an additive log-posterior over candidates:

    log_posterior = chem_logprior              # gated composition prior
                  + iso_weight  * iso_loglik    # erfc, P(envelope|formula)
                  + mass_weight * mass_loglik   # Gaussian ppm, P(precursor|formula)

The prior is a Gaussian Mixture Model over composition features:
    - H/C ratio, O/C ratio           (scale with molecular size)
    - N, S, P, Cl, Br, I counts      (discrete-ish, don't scale with size)
    - RDBE, RDBE / C ratio           (unsaturation)
    - Number of distinct heteroatom types

The GMM captures correlations between features that independent 1D KDEs miss.

Two refinements make the prior behave as a *gate* rather than a ranker:

  1. Two class-specific models.
    -  Halogenated (Cl/Br/I) formulae occupy a different density regime than the rest,
       so each formula is scored against a GMM fit to its own class
       ('halogen' vs 'halofree') and judged against that class's floor.

  2. One-sided soft gating.
    - Each class has a plausibility floor tau (a low percentile of its training
      log-densities).

      log_prior applies a smooth, one-sided penalty anchored at tau:
      ~0 for plausible formulae and increasingly negative below tau,
      so the prior down-ranks the implausible instead of rewarding the typical.
      The transition is a softplus ramp whose steepness (`chem_strength`) and width (`chem_softness`) are live,
      scoring-time knobs. chem_softness=0 means a hard cut-off.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
from molmass import Formula

from . import chem_prior
from ..isotopes.envelope import get_isotope_envelope
from ._coconut_gmm import DEFAULT_GMM_PARAMS
from .likelihoods import isotope_loglik, mass_loglik

if TYPE_CHECKING:
    from ..core.results import FormulaSearchResults, FormulaCandidate
    from ..core.light_formula import LightFormula

def _top_by_posterior(candidates: list, top_n: int | None) -> list:
    """
    The `top_n` candidates by the log-posterior accumulated so far.

    Shared by the isotope and MS2 cascade stages. `None` means no cut. The cut
    is global -- it spans adducts in a concatenated set, so a strong candidate
    under one adduct is not crowded out by weak ones under another.
    """
    if top_n is None or len(candidates) <= top_n:
        return candidates
    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -np.inf if candidates[i].log_posterior is None
            else candidates[i].log_posterior
        ),
        reverse=True,
    )
    return [candidates[i] for i in order[:top_n]]

class FormulaScorer:
    """
    Stacked-likelihood scorer combining:
      - Corpus-derived chemical prior
      - MS1-derived isotope and mass likelihoods
      - MS2-derived fragmentation likelihood

    Example:
        >>> # Use the bundled COCONUT-trained prior
        >>> scorer = FormulaScorer()
        >>> scorer.log_prior(Formula("C6H12O6"))
        -4.21

        >>> # Score a set of candidates against an observed MS1 envelope
        >>> import numpy as np
        >>> observed = np.array([[180.063, 1.0], [181.067, 0.11]])
        >>> scorer.score(results, ms1_peaks=observed, precursor_mz=180.063)
        >>> ranked = results.sort_by_posterior()

        >>> # Or fit your own corpus
        >>> scorer = FormulaScorer().from_corpus(corpus_path="molecular_formulae.txt")
    """

    def __init__(self):
        # chem_prior GMMs
        self.chem_prior_params: chem_prior.GmmParams = chem_prior.from_dict(
            DEFAULT_GMM_PARAMS
        )

        # MS2 reranker, attached via with_ms2()
        self._ms2_model = None

    def with_ms2(
        self,
        model=None,
    ) -> 'FormulaScorer':
        """
        Attach an MS2 reranker so `score()` can populate `ms2_logit`.

        Args:
            model: a `find_mfs.ms2.MistNetNumpy`, or a path to a `.npz`
                artifact exported by mist-fmfs.
                Omit (or pass `None`) to use the reranker weights
                 bundled with find-mfs.

        Returns:
            self, for chaining.

        Example:
            >>> scorer = FormulaScorer().with_ms2()          # bundled weights
            >>> scorer = FormulaScorer().with_ms2("mistnet.npz")  # custom npz
            >>> scorer.score(results, ms2_peaks=peaks, precursor_mz=515.32)
            >>> ranked = results.sort_by_posterior()
        """
        # Imported lazily: find_mfs.ms2 imports back into find_mfs, so a
        # module-level import here would cycle through find_mfs/__init__.
        from ..ms2.net import MistNetNumpy

        if model is None:
            self._ms2_model = MistNetNumpy.default()
        elif isinstance(model, MistNetNumpy):
            self._ms2_model = model
        else:
            self._ms2_model = MistNetNumpy.from_npz(model)
        return self

    def from_corpus(
            self,
            corpus_path: Optional[Path | str] = None,
            corpus_formulae: Optional[Iterable] = None,
            n_components: int = 25,
            random_state: int = 42,
            tau_percentile: float = 0.5,
    ) -> 'FormulaScorer':
        """
        Trains the FormulaScorer's GMMs on a corpus of
         formulae. The GMMs are cached to disk after training,
         so subsequent invocations from the exact same
         corpus file bypass training.

         Corpus can either be given as a filepath (corpus.txt),
         or an iterable of formulae

         Returns:
             self, for chaining
        """
        if corpus_path:
            if not Path(corpus_path).exists():
                raise ValueError(
                    f"{corpus_path} does not exist."
                )

            with open(corpus_path, "r") as f:
                formulae: list[str] = f.readlines()
        elif corpus_formulae:
            formulae: list[str] = list(corpus_formulae)
        else:
            raise ValueError(
                f"Either corpus_path or corpus_formulae must be given"
            )

        self.chem_prior_params = chem_prior.fit(
            formulae,
            n_components,
            random_state,
            tau_percentile,
        )

        return self

    def log_prior(
            self,
            formula: Formula | 'LightFormula',
            strength: float = 1.0,
            softness: float = 1.0,
    ) -> float:
        """
        Chemical-plausibility log prior for a single formula, under this
        scorer's currently loaded `chem_prior_params`. See `score()`'s
        `chem_strength`/`chem_softness` for what `strength`/`softness` do.
        """
        return chem_prior.log_prior(
            self.chem_prior_params,
            formula,
            strength=strength,
            softness=softness,
        )

    @property
    def has_ms2(self) -> bool:
        """
        True once an MS2 reranker has been attached via `with_ms2()`.
        """
        return self._ms2_model is not None

    def score(
        self,
        results: 'FormulaSearchResults',
        *,
        # Chemical Prior
        chem_weight: float = 1.0,
        chem_strength: float = 1.0,
        chem_softness: float = 1.0,
        # Mass log likelihood
        mass_weight: float = 1.0,
        mass_sigma_ppm: float = 5.0,
        # Isotope envelope log likelihood
        precursor_mz: float | None = None,
        ms1_peaks: np.ndarray | None = None,
        iso_weight: float = 1.0,
        iso_ppm: float = 5.0,
        iso_mz_match_da: float = 0.02,
        iso_min_rel: float = 0.02,
        iso_top_n: int | None = 2000,
        # MS2 log likelihood
        ms2_peaks: np.ndarray | None = None,
        instrument: str = 'unknown',
        ms2_top_n: int | None = 256,
        ms2_frag_ppm: float = 10.0,
        ms2_batch_size: int = 256,
    ) -> None:
        """
        Score all candidates in-place with a stacked log-posterior.

        Attaches these scalar log-terms to each candidate:
            - `chem_logprior`: GMM chemical-plausibility prior, log P(formula)
            - `iso_loglik`: isotope-pattern likelihood
                (None when no observed envelope is given or the candidate can't be simulated)
            - `mass_loglik`: mass likelihood (i.e. based on mass error)
            - `ms2_logit`: raw MistNet output
                (None unless an MS2 spectrum was given and a MistNet was
                 attached via `with_ms2()`)

            - `log_posterior`: Combination;
                  chem_weight*chem_logprior
                + iso_weight*iso_loglik
                + mass_weight*mass_loglik

        Note:
            **The MS2 term is not folded into `log_posterior` here**.

            The normalized MS2 term is a softmax over the candidate set,
            so it would go stale whenever results are filtered.

            `score()` stores only the per-mf logits.
            The weighting happens at ranking time:
                scorer.score(results, ms2_peaks=peaks, precursor_mz=mz)
                ranked = results.sort_by_posterior(ms2_weight=1.0)
                # or: results.log_posterior(ms2_weight=1.0)

            See `FormulaCandidate.ms2_logit` and
            `FormulaSearchResults.ms2_loglik()`.

        Args:
            results: FormulaSearchResults to score (mutated in place).
            ms1_peaks: Optional observed MS1 peak list as an (N, 2) array of
                `[m/z, intensity]`. When None, only the prior + mass terms are
                computed and `iso_loglik` stays None.
            precursor_mz: Observed precursor m/z. Currently unused by the
                likelihoods (they anchor on the simulated envelope) but reserved
                for the deferred adduct-partner term. Accepting it now keeps the
                call signature stable.
            mass_sigma_ppm: Sigma (ppm) for the Gaussian precursor-mass term.
            iso_ppm: Mass tolerance (~3-sigma) for the isotope mass term.
            iso_mz_match_da: m/z search half-window for matching predicted peaks.
            iso_min_rel: Predicted peaks below this relative intensity are ignored.
            iso_top_n: Only simulate isotope envelopes for this many candidates,
                chosen by the chem + mass terms. Simulation costs ~0.4 ms each,
                so scoring a full decomposition is seconds per spectrum. None
                scores every candidate. Skipped candidates keep
                `iso_loglik=None` and contribute 0 -- the same as when no MS1
                was supplied.
            iso_weight: Weight on the isotope likelihood
            mass_weight: Weight on the mass likelihood
            chem_weight: Weight on the chemical plausibility prior
            chem_strength: Override for the gate's penalty slope (default: the
                instance's `chem_strength`). Steeper => implausible formulae
                penalized harder.
            chem_softness: Override for the gate's transition width, in units of
                the class's plausibility spread (default: the instance's
                `chem_softness`). 0 => hard hinge; larger => a gentler ramp that
                also nudges borderline formulae below 0.
            ms2_peaks: Optional MS2 peak list as an (N, 2) array of
                `[m/z, intensity]`, already de-isotoped and precursor-cropped.
                Requires a reranker attached via `with_ms2()`; ignored otherwise.
            instrument: Instrument name for the reranker's one-hot encoding.
            ms2_top_n: Only run the reranker on this many candidates, chosen by
                the cheap terms (chem + mass + iso). MS2 costs ~O(candidates)
                network evaluations with an O(peaks^2) constant, so scoring a
                full decomposition is seconds per spectrum. None scores every
                candidate. Candidates outside the cut keep `ms2_logit=None` and
                are floored -- never favoured -- by `ms2_loglik()`.

                Note this makes the pipeline a **cascade**, not a pure additive
                posterior: a candidate the cheap terms rank poorly is never
                given the chance to be rescued by MS2 evidence.
            ms2_frag_ppm: Fragment mass tolerance for subformula assignment.
            ms2_batch_size: Candidates per reranker forward pass.

        Raises:
            ValueError: If `ms2_peaks` is given without a reranker attached.
        """
        # Materialize once: `results.candidates` rebuilds the list on every
        # access (the underlying FormulaCandidates are cached, so mutating
        # these objects does persist).
        candidates = results.candidates
        n_candidates = len(candidates)

        # Calculate chem log priors
        chem_logpriors: np.ndarray = chem_prior.batch_log_prior(
            gmm_params=self.chem_prior_params,
            formulae=[c.formula for c in candidates],
            strength=chem_strength,
            softness=chem_softness,
        )

        # Calculate mass log liks
        mass_logliks: np.ndarray = np.array([
            mass_loglik(
                candidate.error_ppm,
                sigma_ppm=mass_sigma_ppm,
            ) for candidate in candidates
        ])

        # Calculate isotope log_lik
        # IsoSpecPy calls are ~0.4 ms per candidate, so only the top
        # `iso_top_n` candidates are used, sorted by semi_log_posterior
        iso_logliks: np.ndarray = np.full(
            len(candidates),
            fill_value=None,
        )
        if ms1_peaks is not None:
            semi_log_posterior: np.ndarray = (
                    chem_weight * chem_logpriors
                    + mass_weight * mass_logliks
            )

            if iso_top_n is None or iso_top_n >= n_candidates:
                top_idx = np.arange(n_candidates)
            else:
                top_idx: np.ndarray = np.argpartition(
                    -semi_log_posterior, iso_top_n
                )[:iso_top_n]

            top_logliks = np.array([
                isotope_loglik(
                    ion_formula=candidates[i].ion_formula,
                    ms1_peaks=ms1_peaks,
                    ppm=iso_ppm,
                    mz_match_da=iso_mz_match_da,
                    min_rel=iso_min_rel,
                ) for i in top_idx
            ])

            # For candidates in which weren't scored,
            #   give whatever the minimum iso_loglik was
            iso_logliks: np.ndarray = np.full(
                len(candidates),
                fill_value=min(top_logliks),
                dtype=np.float64,
            )
            iso_logliks[top_idx] = top_logliks

        # Finally assign chem_log_prior, mass_loglik, and iso_loglik to each
        for i, candidate in enumerate(candidates):
            candidate.chem_logprior = float(chem_logpriors[i])
            candidate.mass_loglik = float(mass_logliks[i])
            candidate.iso_loglik = iso_logliks[i]

            candidate.log_posterior = (
                    chem_weight * candidate.chem_logprior
                    + mass_weight * candidate.mass_loglik
                    + iso_weight * (
                        0.0 if candidate.iso_loglik is None
                        else candidate.iso_loglik
                    )
            )

        if ms2_peaks is not None:
            self._score_ms2(
                results, candidates, ms2_peaks,
                precursor_mz=precursor_mz,
                instrument=instrument,
                top_n=ms2_top_n,
                frag_ppm=ms2_frag_ppm,
                batch_size=ms2_batch_size,
            )

    def _score_ms2(
        self,
        results: 'FormulaSearchResults',
        candidates: list,
        ms2_peaks: np.ndarray,
        *,
        precursor_mz: float | None,
        instrument: str,
        top_n: int | None,
        frag_ppm: float,
        batch_size: int,
    ) -> None:
        """
        Populate `ms2_logit` on the top candidates

        Runs only on the `top_n` candidates ranked by log_posterior,
         because MistNet relatively expensive to run.

        The rest keep `ms2_logit = None`.
        """
        if self._ms2_model is None:
            raise ValueError(
                "ms2_peaks was given but no MistNet is attached. "
                "Call scorer.with_ms2(model_or_npz_path) first."
            )
        if not candidates:
            return

        from ..ms2 import ms2_logits, resolve_ion

        if precursor_mz is None:
            precursor_mz = results.query_mass

        # Rank by the cheap terms to choose who is worth the network pass.
        # The cut is global: it spans adducts, so a strong candidate under one
        # adduct is not crowded out by weak ones under another.
        selected = _top_by_posterior(candidates, top_n)

        # Group by ion: the reranker takes one ion per call (it drives both the
        # subformula assignment and the ion one-hot). Candidates whose ion is
        # outside the reranker's fixed positive-mode vocabulary -- including all
        # of negative mode -- are skipped and keep ms2_logit = None.
        by_ion: dict[str, list] = {}
        for candidate in selected:
            charge = (
                candidate.ion_formula.charge
                if candidate.ion_formula is not None else 1
            )
            ion = resolve_ion(candidate.adduct, charge)
            if ion is not None:
                by_ion.setdefault(ion, []).append(candidate)

        for ion, group in by_ion.items():
            logits = ms2_logits(
                self._ms2_model,
                [c.formula.formula for c in group],
                ion,
                ms2_peaks,
                precursor_mz=precursor_mz,
                instrument=instrument,
                frag_ppm=frag_ppm,
                batch_size=batch_size,
            )
            for candidate, logit in zip(group, logits):
                candidate.ms2_logit = float(logit)
