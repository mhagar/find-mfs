"""
Bayesian formula prior using a Gaussian Mixture Model trained on
a corpus of known molecular formulae (e.g. COCONUT, NPAtlas).

Models P(formula) as a joint distribution over composition features:
    - H/C ratio, O/C ratio           (scale with molecular size)
    - N, S, P, Cl, Br, I counts      (discrete-ish, don't scale with size)
    - RDBE, RDBE / C ratio            (unsaturation)
    - Number of distinct heteroatom types

The GMM captures correlations between features that independent
1D KDEs miss (e.g. high N + high O is plausible for amino acids,
but high N + high S is rare).
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

if TYPE_CHECKING:
    from ..core.finder import FormulaCandidate
    from ..core.results import FormulaSearchResults

# Feature layout — order matters, keep in sync with _formula_to_features()
_RATIO_FEATURES = ('H', 'O')           # modelled as X/C
_COUNT_FEATURES = ('N', 'S', 'P', 'Cl', 'Br', 'I')  # raw counts
# + RDBE, RDBE/C, n_heteroatom_types  (3 extra)
_N_FEATURES = len(_RATIO_FEATURES) + len(_COUNT_FEATURES) + 3

# Small floor to prevent -inf scores
_LOG_FLOOR = -50.0


def _rdbe(counts: dict[str, int]) -> float:
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


def _formula_to_features(elem_counts: dict[str, int]) -> np.ndarray | None:
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
    """Extract element counts from a Formula or LightFormula."""
    counts: dict[str, int] = {}
    comp = formula.composition()
    for symbol, item in comp.items():
        if symbol == '' or symbol == 'e-':
            continue
        if item.count > 0:
            counts[symbol] = item.count
    return counts


class FormulaPrior:
    """
    Corpus-derived prior P(formula) using a Gaussian Mixture Model
    over molecular composition features.

    Example:
        >>> corpus = Path("COCONUT_molecular_formulae.txt").read_text().splitlines()
        >>> prior = FormulaPrior().fit(corpus)
        >>> prior.log_prior(Formula("C6H12O6"))
        -4.21
    """

    def __init__(self):
        self._gmm: GaussianMixture | None = None
        self._fitted = False

    def fit(
        self,
        formulae: list[str],
        n_components: int = 25,
        random_state: int = 42,
    ) -> 'FormulaPrior':
        """
        Fit the GMM on a corpus of molecular formula strings.

        Caches the fitted model on disk keyed by (corpus hash, n_components,
        random_state) — subsequent calls with the same args load instantly.

        Args:
            formulae: List of formula strings (e.g. ["C6H12O6", ...])
            n_components: Number of Gaussian components (use select_n_components()
                          to choose via BIC if unsure).
            random_state: Random seed for reproducibility.

        Returns:
            self, for chaining.
        """
        cache_dir = self._CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        gmm_cache = (
            cache_dir
            / f'gmm_{self._corpus_hash(formulae)}_k{n_components}_rs{random_state}.json'
        )

        if gmm_cache.exists():
            self.load(gmm_cache)
            return self

        features = self._parse_corpus(formulae)

        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state,
            n_init=3,
        )
        self._gmm.fit(features)
        self._fitted = True

        self.save(gmm_cache)
        return self

    _CACHE_DIR = Path(__file__).resolve().parent / '.cache'

    @staticmethod
    def _corpus_hash(formulae: list[str]) -> str:
        """Deterministic hash of the corpus content."""
        h = hashlib.sha256()
        for s in formulae:
            h.update(s.encode())
        return h.hexdigest()[:16]

    @classmethod
    def _parse_corpus(cls, formulae: list[str]) -> np.ndarray:
        """Parse formula strings into feature matrix, with disk caching."""
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
        Compute log P(formula) under the fitted GMM prior.

        Args:
            formula: A molmass.Formula or LightFormula instance.

        Returns:
            Log-probability score (higher = more plausible).
            Returns 0.0 for formulae without carbon.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before log_prior()")

        elem_counts = _get_element_counts(formula)
        feat = _formula_to_features(elem_counts)

        if feat is None:
            return 0.0

        score = float(self._gmm.score_samples(feat.reshape(1, -1))[0])
        return max(score, _LOG_FLOOR)

    def score_results(
        self,
        results: 'FormulaSearchResults',
        mass_sigma_ppm: float,
        isotope_sigma: float,
    ) -> None:
        """
        Score all candidates using the full posterior (in-place).

        Computes:
            log P(formula | data) = log P(prior)
                                   - Δm² / (2 * mass_sigma_ppm²)
                                   - RMSE² / (2 * isotope_sigma²)  [if available]
        """
        for candidate in results.candidates:
            candidate: 'FormulaCandidate'
            log_posterior = self.log_prior(candidate.formula)
            candidate.prior_score = log_posterior

            # Mass error likelihood
            log_posterior -= (
                candidate.error_ppm ** 2 /
                (2 * mass_sigma_ppm ** 2)
            )

            # Isotope likelihood (if available)
            if candidate.isotope_match_result is not None:
                rmse = candidate.isotope_match_result.intensity_rmse
                log_posterior -= rmse ** 2 / (2 * isotope_sigma ** 2)

            candidate.posterior_score = log_posterior

    def save(self, path: Path | str) -> None:
        """Save fitted GMM parameters to a JSON file."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before save()")

        path = Path(path)
        data = {
            'n_components': self._gmm.n_components,
            'weights': self._gmm.weights_.tolist(),
            'means': self._gmm.means_.tolist(),
            'covariances': self._gmm.covariances_.tolist(),
        }
        path.write_text(json.dumps(data))

    def load(self, path: Path | str) -> 'FormulaPrior':
        """Load GMM parameters from a JSON file (no sklearn needed at runtime)."""
        path = Path(path)
        data = json.loads(path.read_text())

        self._gmm = GaussianMixture(
            n_components=data['n_components'],
            covariance_type='full',
        )
        # Inject fitted parameters directly
        self._gmm.weights_ = np.array(data['weights'])
        self._gmm.means_ = np.array(data['means'])
        self._gmm.covariances_ = np.array(data['covariances'])
        self._gmm.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(self._gmm.covariances_)
        )
        self._fitted = True
        return self
