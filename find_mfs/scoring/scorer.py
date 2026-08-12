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

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from molmass import Formula
from sklearn.mixture import GaussianMixture

from ..core.light_formula import LightFormula
from ..isotopes.envelope import get_isotope_envelope
from ._coconut_gmm import DEFAULT_GMM_PARAMS
from .likelihoods import isotope_loglik, mass_loglik

if TYPE_CHECKING:
    from ..core.results import FormulaSearchResults

# Feature layout — order matters, keep in sync with _formula_to_features()
_RATIO_FEATURES = ('H', 'O')           # modelled as X/C
_COUNT_FEATURES = ('N', 'S', 'P', 'Cl', 'Br', 'I')  # raw counts
# + RDBE, RDBE/C, n_heteroatom_types  (3 extra)
_N_FEATURES = len(_RATIO_FEATURES) + len(_COUNT_FEATURES) + 3

# The prior is split into two composition-class models, each judged against its
# own kind. A formula routes to 'halogen' if it contains any of these elements,
# else to 'halofree' (fluorine deliberately stays with 'halofree').
_ROUTING_HALOGENS = ('Cl', 'Br', 'I')
_CLASSES = ('halofree', 'halogen')
# Feature-matrix column indices of the routing halogens, derived from the layout
# so a class split can be done directly on the (cached) feature matrix.
_HALOGEN_COLS = tuple(
    len(_RATIO_FEATURES) + _COUNT_FEATURES.index(h) for h in _ROUTING_HALOGENS
)

# Small floor to prevent -inf scores
_LOG_FLOOR = -50.0

# Default live knobs for the soft plausibility gate (see _soft_gate / log_prior).
_DEFAULT_CHEM_STRENGTH = 1.0   # penalty slope in nats per nat of log-density below tau
_DEFAULT_CHEM_SOFTNESS = 1.0   # transition width in units of the class's density
                               # spread (scale); 0.0 => hard hinge


def _soft_gate(
        raw: float,
        tau: float,
        strength: float,
        softness: float,
) -> float:
    """
    Smooth one-sided plausibility penalty anchored at the floor `tau`.

    Returns ~0 for a plausible formula (raw >> tau) and a negative penalty that
    grows with implausibility (raw << tau), floored at `_LOG_FLOOR`:

        penalty = -strength * softness * softplus(-(raw - tau) / softness)

    which is a softplus ramp of width ~`softness` (in nats of log-density) around
    tau with asymptotic slope `strength`. As `softness -> 0` this collapses to the
    hard hinge `strength * min(raw - tau, 0)`. Both knobs are applied at scoring
    time, so they can be tuned live without refitting the GMM.
    """
    d = raw - tau
    if softness <= 0.0:
        penalty = strength * min(d, 0.0)
    else:
        # softplus(x) = max(x, 0) + log1p(exp(-|x|)), numerically stable.
        x = -d / softness
        softplus = max(x, 0.0) + math.log1p(math.exp(-abs(x)))
        penalty = -strength * softness * softplus
    return max(penalty, _LOG_FLOOR)

# Envelope simulation resolution for isotope scoring.
_SIM_MZ_TOLERANCE = 0.05
_SIM_INTENSITY_THRESHOLD = 0.001


def _rdbe(
        counts: dict[str, int],
) -> float:
    """
    Ring and double bond equivalence.
    RDBE = 1 + C - H/2 + N/2 + P/2 - (Cl + Br + I + F)/2
    """
    c = counts.get('C', 0)
    h = counts.get('H', 0)
    n = counts.get('N', 0)
    p = counts.get('P', 0)
    hal = (
        counts.get('F', 0) + counts.get('Cl', 0)
        + counts.get('Br', 0) + counts.get('I', 0)
    )
    return 1.0 + c - h / 2.0 + n / 2.0 + p / 2.0 - hal / 2.0


_HETEROATOMS = {'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Se', 'Si', 'B'}


def _formula_to_features(
        elem_counts: dict[str, int],
) -> np.ndarray | None:
    """
    Convert element counts dict to the feature vector.
    Returns None if the formula has no carbon (can't compute ratios).
    """
    c_count = elem_counts.get('C', 0)
    if c_count == 0:
        return None

    feats = np.empty(_N_FEATURES, dtype=np.float64)
    i = 0

    # Ratio features (X / C)
    for elem in _RATIO_FEATURES:
        feats[i] = elem_counts.get(elem, 0) / c_count
        i += 1

    # Count features
    for elem in _COUNT_FEATURES:
        feats[i] = elem_counts.get(elem, 0)
        i += 1

    # RDBE
    rdbe = _rdbe(elem_counts)
    feats[i] = rdbe
    i += 1

    # RDBE / C
    feats[i] = rdbe / c_count
    i += 1

    # Number of distinct heteroatom types
    feats[i] = sum(1 for e in _HETEROATOMS if elem_counts.get(e, 0) > 0)

    return feats


def _get_element_counts(formula) -> dict[str, int]:
    """
    Extract element counts from a Formula or LightFormula
    """
    counts: dict[str, int] = {}
    comp = formula.composition()
    for symbol, item in comp.items():
        if symbol == '' or symbol == 'e-':
            continue
        if item.count > 0:
            counts[symbol] = item.count
    return counts


def _is_halogenated(
        counts: dict[str, int],
) -> bool:
    """True if the formula contains any routing halogen (Cl/Br/I)."""
    return any(counts.get(h, 0) > 0 for h in _ROUTING_HALOGENS)


def _select_class(
        counts: dict[str, int],
) -> str:
    """Route a formula to its composition-class model ('halogen' or 'halofree')."""
    return 'halogen' if _is_halogenated(counts) else 'halofree'


class FormulaScorer:
    """
    Stacked-likelihood scorer combining a corpus-derived chemical prior with
    MS1 isotope and mass likelihoods.

    Example:
        >>> # Use the bundled COCONUT-trained prior
        >>> scorer = FormulaScorer.default()
        >>> scorer.log_prior(Formula("C6H12O6"))
        -4.21

        >>> # Score a set of candidates against an observed MS1 envelope
        >>> import numpy as np
        >>> observed = np.array([[180.063, 1.0], [181.067, 0.11]])
        >>> scorer.score(results, ms1_peaks=observed, precursor_mz=180.063)
        >>> ranked = results.sort_by_posterior()

        >>> # Or fit your own corpus
        >>> corpus = Path("molecular_formulae.txt").read_text().splitlines()
        >>> scorer = FormulaScorer().fit(corpus)
    """

    def __init__(self):
        # One GMM + plausibility floor (tau) per composition class. A class stays
        # None when the corpus had too few formulae of that kind to fit.
        self._models: dict[str, GaussianMixture | None] = {
            'halofree': None, 'halogen': None,
        }
        self._taus: dict[str, float | None] = {
            'halofree': None, 'halogen': None,
        }
        # Per-class spread of training log-densities; the unit for `chem_softness`.
        self._scales: dict[str, float | None] = {
            'halofree': None, 'halogen': None,
        }
        # Live knobs for the soft plausibility gate. Tune these directly on the
        # instance (e.g. `scorer.chem_softness = 5`) — they take effect on the
        # next score()/log_prior() call, no refit required.
        self.chem_strength: float = _DEFAULT_CHEM_STRENGTH
        self.chem_softness: float = _DEFAULT_CHEM_SOFTNESS
        # MS2 reranker, attached via with_ms2(). None => no MS2 term.
        self._ms2_model = None

    def with_ms2(
        self,
        model,
    ) -> 'FormulaScorer':
        """
        Attach an MS2 reranker so `score()` can populate `ms2_logit`.

        Args:
            model: a `find_mfs.ms2.MistNetNumpy`, or a path to the `.npz`
                artifact exported by mist-fmfs.

        Returns:
            self, for chaining.

        Example:
            >>> scorer = FormulaScorer.default().with_ms2("mistnet.npz")
            >>> scorer.score(results, ms2_peaks=peaks, precursor_mz=515.32)
            >>> ranked = results.sort_by_posterior()
        """
        # Imported lazily: find_mfs.ms2 imports back into find_mfs, so a
        # module-level import here would cycle through find_mfs/__init__.
        from ..ms2.net import MistNetNumpy

        self._ms2_model = (
            model if isinstance(model, MistNetNumpy) else MistNetNumpy.from_npz(model)
        )
        return self

    @property
    def has_ms2(self) -> bool:
        """True once an MS2 reranker has been attached via `with_ms2()`."""
        return self._ms2_model is not None

    @property
    def _fitted(self) -> bool:
        """True once at least one composition-class model is available."""
        return any(m is not None for m in self._models.values())

    def fit(
        self,
        formulae: list[str],
        n_components: int = 25,
        random_state: int = 42,
        tau_percentile: float = 0.5,
    ) -> 'FormulaScorer':
        """
        Fit the per-class GMMs on a corpus of molecular formula strings.

        The corpus is split into composition classes (halogenated vs not) and a
        separate GMM is fit for each. Each class also gets a plausibility floor
        `tau`, the `tau_percentile`-th percentile of that class's own training
        log-densities, used by `log_prior` to gate implausible formulae.

        A class with too few formulae to fit (< max(n_components, 10)) is simply
        skipped (its model stays None); as long as one class fits, this succeeds.

        Caches the fitted models on disk keyed by
        (corpus hash, n_components, random_state, tau_percentile) so that
        subsequent calls with the same args load instantly.

        Args:
            formulae: List of formula strings (e.g. ["C6H12O6", ...])
            n_components: Number of Gaussian components per class (use
                          select_n_components() to choose via BIC if unsure).
            random_state: Random seed for reproducibility.
            tau_percentile: Percentile of each class's training log-densities used
                            as its plausibility floor. Lower = gentler gate.

        Returns:
            self, for chaining.
        """
        cache_dir = self._CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        gmm_cache = (
            cache_dir
            / f'gmm2_{self._corpus_hash(formulae)}_k{n_components}'
              f'_rs{random_state}_p{tau_percentile}.json'
        )

        if gmm_cache.exists():
            self.load(gmm_cache)
            return self

        features = self._parse_corpus(formulae)
        # Split on the routing-halogen columns directly (no re-parse needed).
        halo_mask = features[:, list(_HALOGEN_COLS)].sum(axis=1) > 0
        splits = {'halofree': features[~halo_mask], 'halogen': features[halo_mask]}

        self._models = {'halofree': None, 'halogen': None}
        self._taus = {'halofree': None, 'halogen': None}
        self._scales = {'halofree': None, 'halogen': None}
        for cls in _CLASSES:
            X = splits[cls]
            if X.shape[0] < max(n_components, 10):
                continue  # too few examples of this class — skip it
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=random_state,
                n_init=3,
            )
            gmm.fit(X)
            train_scores = gmm.score_samples(X)
            self._models[cls] = gmm
            self._taus[cls] = float(np.percentile(train_scores, tau_percentile))
            # Spread of plausibility among real compounds — the softness unit.
            self._scales[cls] = float(np.std(train_scores)) or 1.0

        if not self._fitted:
            raise ValueError(
                "No composition class had enough formulae to fit "
                f"(need >= max(n_components={n_components}, 10) per class)."
            )

        self.save(gmm_cache)
        return self

    _CACHE_DIR = Path(__file__).resolve().parent / '.cache'

    @classmethod
    def default(cls) -> 'FormulaScorer':
        """
        Load the scorer bundled with find-mfs (chemical prior trained on
        formulae pulled from COCONUT).

        Returns:
            A ready-to-use FormulaScorer instance.
        """
        return cls()._load_params(DEFAULT_GMM_PARAMS)

    @staticmethod
    def _corpus_hash(formulae: list[str]) -> str:
        """
        Return a hash of the corpus content
        """
        h = hashlib.sha256()
        for s in formulae:
            h.update(s.encode())
        return h.hexdigest()[:16]

    @classmethod
    def _parse_corpus(
            cls,
            formulae: list[str]
    ) -> np.ndarray:
        """
        Parse formula strings into feature matrix, with disk caching
        (i.e. skips parsing if already cached)
        """
        cache_dir = cls._CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f'features_{cls._corpus_hash(formulae)}.npy'

        if cache_path.exists():
            return np.load(cache_path)

        rows = []
        for formula_str in formulae:
            formula_str = formula_str.strip()
            if not formula_str:
                continue
            try:
                f = Formula(formula_str)
                counts = _get_element_counts(f)
            except Exception:
                continue

            feat = _formula_to_features(counts)
            if feat is not None:
                rows.append(feat)

        if len(rows) < 10:
            raise ValueError(
                f"Only {len(rows)} valid formulae parsed — need more data"
            )

        features = np.array(rows)
        np.save(cache_path, features)
        return features

    def select_n_components(
        self,
        formulae: list[str],
        candidates: list[int] | None = None,
        random_state: int = 42,
    ) -> dict[int, float]:
        """
        Fit GMMs with different component counts and return BIC scores.

        Can be used to find ideal component count

        Args:
            formulae: Corpus of formula strings.
            candidates: List of n_components values to try.
                Defaults to [5, 10, 15, 20, 25, 30, 40, 50].
            random_state: Random seed.

        Returns:
            Dict mapping n_components -> BIC score (lower is better).
        """
        if candidates is None:
            candidates = [5, 10, 15, 20, 25, 30, 40, 50]

        features = self._parse_corpus(formulae)
        bics = {}

        for k in candidates:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=random_state,
                n_init=3,
            )
            gmm.fit(features)
            bics[k] = gmm.bic(features)

        return bics

    def log_prior(
        self,
        formula: Formula | LightFormula,
    ) -> float:
        """
        Compute the gated chemical-plausibility log-prior for a formula.

        The formula is routed to its composition-class GMM, scored, and passed
        through a smooth one-sided gate anchored at that class's plausibility floor
        tau (see _soft_gate): ~0 for a plausible formula — the prior stays out of
        the way and lets the mass/isotope likelihoods decide — and increasingly
        negative for an implausible one (down to _LOG_FLOOR). The gate's steepness
        and width come from the live `chem_strength` / `chem_softness` knobs. If a
        formula's class has no fitted model, it falls back to the other class.

        Args:
            formula: A molmass.Formula or LightFormula instance.

        Returns:
            A gated log-prior in [_LOG_FLOOR, 0.0]. Returns 0.0 (neutral) for
            formulae without carbon.
        """
        return self._gated_log_prior(
            formula, self.chem_strength, self.chem_softness
        )

    def _gated_log_prior(
        self,
        formula: Formula | LightFormula,
        strength: float,
        softness: float,
    ) -> float:
        """
        Route a formula to its class GMM and apply the soft gate with the given
        knobs.

        Shared by log_prior() (instance knobs) and score() (call overrides)
        """
        if not self._fitted:
            raise ValueError(
                "GMM not yet trained/loaded. "
                "Instantiate using FormulaScorer.default() to use a GMM pre-fit to COCONUT. "
                "Otherwise, use scorer.fit() or .load() first."
            )

        elem_counts = _get_element_counts(formula)
        feat = _formula_to_features(elem_counts)

        if feat is None:
            return 0.0

        cls = _select_class(elem_counts)
        gmm, tau, scale = self._models[cls], self._taus[cls], self._scales[cls]
        if gmm is None:
            # This class wasn't fitted — fall back to the other one.
            other = 'halogen' if cls == 'halofree' else 'halofree'
            gmm, tau, scale = (
                self._models[other], self._taus[other], self._scales[other]
            )

        raw = float(gmm.score_samples(feat.reshape(1, -1))[0])
        # `softness` is in class-spread units; convert to nats for the gate.
        return _soft_gate(raw, tau, strength, softness * (scale or 1.0))

    def _batch_log_prior(
        self,
        formulae: list,
        strength: float,
        softness: float,
    ) -> np.ndarray:
        """
        Vectorized `_gated_log_prior` over many formulae.

        Identical results to calling `_gated_log_prior` per formula, but issues
        one `GaussianMixture.score_samples` call per composition class instead of
        one per formula. sklearn's per-call overhead dominates at this feature
        size, so batching is ~175x faster -- which is the difference between the
        prior being free and it being the pipeline's bottleneck.
        """
        if not self._fitted:
            raise ValueError(
                "GMM not yet trained/loaded. "
                "Instantiate using FormulaScorer.default() to use a GMM pre-fit to COCONUT. "
                "Otherwise, use scorer.fit() or .load() first."
            )

        n = len(formulae)
        out = np.zeros(n, dtype=np.float64)
        if n == 0:
            return out

        # Bucket by composition class. Formulae with no carbon cannot be
        # featurized and stay at 0.0, matching the scalar path.
        rows: dict[str, list[int]] = {cls: [] for cls in _CLASSES}
        feats: dict[str, list[np.ndarray]] = {cls: [] for cls in _CLASSES}
        for i, formula in enumerate(formulae):
            elem_counts = _get_element_counts(formula)
            feat = _formula_to_features(elem_counts)
            if feat is None:
                continue
            cls = _select_class(elem_counts)
            rows[cls].append(i)
            feats[cls].append(feat)

        for cls in _CLASSES:
            if not rows[cls]:
                continue
            gmm, tau, scale = self._models[cls], self._taus[cls], self._scales[cls]
            if gmm is None:
                other = 'halogen' if cls == 'halofree' else 'halofree'
                gmm, tau, scale = (
                    self._models[other], self._taus[other], self._scales[other]
                )

            raw = gmm.score_samples(np.vstack(feats[cls]))
            out[rows[cls]] = _soft_gate_array(
                raw, tau, strength, softness * (scale or 1.0)
            )
        return out

    def score(
        self,
        results: 'FormulaSearchResults',
        *,
        ms1_peaks: np.ndarray | None = None,
        precursor_mz: float | None = None,
        mass_sigma_ppm: float = 5.0,
        iso_ppm: float = 5.0,
        iso_mz_match_da: float = 0.02,
        iso_min_rel: float = 0.02,
        iso_weight: float = 1.0,
        mass_weight: float = 1.0,
        chem_weight: float = 1.0,
        chem_strength: float | None = None,
        chem_softness: float | None = None,
        ms2_peaks: np.ndarray | None = None,
        instrument: str = 'unknown',
        ms2_top_n: int | None = 256,
        ms2_frag_ppm: float = 10.0,
        ms2_use_halogens: bool = True,
        ms2_batch_size: int = 256,
    ) -> None:
        """
        Score all candidates in-place with a stacked log-posterior.

        Attaches these scalar log-terms to each candidate:
            - `chem_logprior`: GMM chemical-plausibility prior, log P(formula)
            - `iso_loglik`: isotope-pattern likelihood (None when no
              observed envelope is given or the candidate can't be simulated)
            - `mass_loglik`: mass likelihood (i.e. based on mass error)
            - `ms2_logit`: raw MS2 reranker output (None unless an MS2 spectrum
              was given and a reranker was attached via `with_ms2()`)
            - `log_posterior`:
                chem_weight*chem_logprior
                + iso_weight*iso_loglik  (missing iso term contributes 0)
                + mass_weight*mass_loglik

        Candidates are never dropped: a poor isotope match yields a low
        `iso_loglik`, not an omission.

        Note:
            **The MS2 term is not folded into `log_posterior` here**, and there
            is deliberately no `ms2_weight` argument on this method. The
            normalized MS2 term is a softmax over the candidate set, so it only
            has meaning relative to a particular set and would go stale the
            moment the results were filtered. `score()` therefore stores only
            the per-candidate logit; the weighting happens at ranking time:

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
            ms2_use_halogens: Search CHNOPS+FClBrI when assigning subformulae.
                Should match the element set the candidates were found with.
            ms2_batch_size: Candidates per reranker forward pass.

        Raises:
            ValueError: If `ms2_peaks` is given without a reranker attached.
        """
        obs = np.asarray(ms1_peaks, dtype=float) if ms1_peaks is not None else None
        strength = self.chem_strength if chem_strength is None else chem_strength
        softness = self.chem_softness if chem_softness is None else chem_softness

        # Materialize once: `results.candidates` rebuilds the list on every
        # access (the underlying FormulaCandidates are cached, so mutating
        # these objects does persist).
        candidates = results.candidates

        # One GMM call per composition class rather than one per candidate.
        chem_logpriors = self._batch_log_prior(
            [c.formula for c in candidates], strength, softness
        )

        for candidate, chem_logprior in zip(candidates, chem_logpriors):
            chem_logprior = float(chem_logprior)
            candidate.chem_logprior = chem_logprior

            m_loglik = mass_loglik(candidate.error_ppm, sigma_ppm=mass_sigma_ppm)
            candidate.mass_loglik = m_loglik

            i_loglik = None
            if obs is not None:
                ion_formula = candidate.ion_formula or candidate.formula
                if ion_formula is not None:
                    pred_env = get_isotope_envelope(
                        formula=ion_formula,
                        mz_tolerance=_SIM_MZ_TOLERANCE,
                        threshold=_SIM_INTENSITY_THRESHOLD,
                    )
                    i_loglik = isotope_loglik(
                        pred_env,
                        obs,
                        ppm=iso_ppm,
                        mz_match_da=iso_mz_match_da,
                        min_rel=iso_min_rel,
                    )
            candidate.iso_loglik = i_loglik

            log_posterior = chem_weight * chem_logprior + mass_weight * m_loglik
            if i_loglik is not None:
                log_posterior += iso_weight * i_loglik
            candidate.log_posterior = log_posterior

    def save(self, path: Path | str) -> None:
        """
        Save fitted per-class GMM parameters to a JSON file.

        The layout is one block per composition class (None for a class that
        wasn't fitted): {"halofree": {...} | None, "halogen": {...} | None}.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before save()")

        path = Path(path)
        data: dict[str, dict | None] = {}
        for cls in _CLASSES:
            gmm = self._models[cls]
            if gmm is None:
                data[cls] = None
                continue
            data[cls] = {
                'n_components': gmm.n_components,
                'weights': gmm.weights_.tolist(),
                'means': gmm.means_.tolist(),
                'covariances': gmm.covariances_.tolist(),
                'tau': self._taus[cls],
                'scale': self._scales[cls],
            }
        path.write_text(json.dumps(data))

    def load(self, path: Path | str) -> 'FormulaScorer':
        """
        Load per-class GMM parameters from a JSON file (no sklearn fitting needed)
        """
        data = json.loads(Path(path).read_text())
        return self._load_params(data)

    def _load_params(self, data: dict) -> 'FormulaScorer':
        """
        Inject per-class GMM parameters from a dict (JSON cache or bundled module).

        Expects the nested {class: block | None} layout produced by save().
        """
        self._models = {'halofree': None, 'halogen': None}
        self._taus = {'halofree': None, 'halogen': None}
        self._scales = {'halofree': None, 'halogen': None}
        for cls in _CLASSES:
            block = data.get(cls)
            if block is None:
                continue
            gmm = GaussianMixture(
                n_components=block['n_components'],
                covariance_type='full',
            )
            # Inject fitted parameters directly
            gmm.weights_ = np.array(block['weights'])
            gmm.means_ = np.array(block['means'])
            gmm.covariances_ = np.array(block['covariances'])
            gmm.precisions_cholesky_ = np.linalg.cholesky(
                np.linalg.inv(gmm.covariances_)
            )
            self._models[cls] = gmm
            self._taus[cls] = block['tau']
            # `scale` may be absent in legacy params — fall back to nat-units.
            self._scales[cls] = block.get('scale') or 1.0

        if not self._fitted:
            raise ValueError("Loaded parameters contain no fitted model.")
        return self
