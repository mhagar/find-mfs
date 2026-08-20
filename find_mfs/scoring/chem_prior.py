"""
An estimation of a formula's 'chemical plausibility' using
GMMs trained on a corpus of formula strings (i.e.
from COCONUT DB by default)

This estimate is used as a prior when scoring
a find-mfs query
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from molmass import Formula
from sklearn.mixture import GaussianMixture

from ..core.light_formula import LightFormula
from .featurization import (
    featurize_formula, contains_halogen, split_matrix_by_halogen
)

# Dir to save/load cached GMMs
_CACHE_DIR = Path(__file__).resolve().parent / '.cache'

# Bump whenever what save()/load() persist (or fit()'s behavior on a too-thin
# class) changes. Baked into the cache filename so an old-format cache file is
# simply invisible to a new fit() call -- never silently reused as if it were
# still valid -- rather than needing per-field migration logic.
_CACHE_SCHEMA_VERSION = 1

# Minimum log prior floor
_LOG_FLOOR = -50.0

# Two different GMMs, one or halofree and one for halogen
# Note: fluorine stays w halofree
CLASSES = ('halofree', 'halogen')
# GmmParams TypedDict must match!
class GmmParams(TypedDict):
    halofree: tuple[GaussianMixture, float, float]
    halogen: tuple[GaussianMixture, float, float]

#### INFERENCE ####
def log_prior(
        gmm_params: GmmParams,
        formula: Formula | LightFormula,
        strength: float = 1.0,
        softness: float = 1.0,
) -> float:
    """
    Estimate chemical plausibility for a formula, used
    as a log prior.

    1. The formula is featurized
    2. Then routed to either the halofree or halogenated GMM
    3. The raw log density is calculated and then 'softly chopped'
        such that the maximum score a formula can receive is ~0
        - This is so the term acts as a penalty against implausible formulae,
            rather than as a reward for *common* formulae
    4. The boundary/cut-off is smoothed according to
        `strength`/`softness`

    Args:
        gmm_params:
        formula: A molmass.Formula or LightFormula instance.
        softness: penalty slope in nats per nat of log-density below tau
            (default 1.0)
        strength: transition width in units of the class's density.
            0 = sharp boundary (default 1.0)

    Returns:
        A float [_LOG_FLOOR, 0.0]
        Returns _LOG_FLOOR for formulae without carbon
    """
    feat = featurize_formula(formula)
    if feat is None:
        return _LOG_FLOOR

    # Toggle bw halo/halo-free GMMs depending on formula
    cls: Literal['halogen', 'halofree'] = _select_gmm_class(feat)
    gmm, tau, scale = gmm_params[cls]

    raw = float(
        gmm.score_samples(feat.reshape(1, -1))[0]
    )

    # `softness` is in class-spread units; convert to nats for the gate.
    return _soft_gate(
        raw,
        tau,
        strength,
        softness * scale
    )


def batch_log_prior(
        gmm_params: GmmParams,
        formulae: list[Formula | LightFormula],
        strength: float = 1.0,
        softness: float = 1.0,
) -> np.ndarray:
    """
    Vectorized `log_prior` over many formulae

    Identical results to calling `log_prior` per formula,
    but issues one `GaussianMixture.score_samples` call per GMM
    instead of one per formula.

    sklearn's per-call overhead dominates at this feature size,
    so batching is faster
    """
    n = len(formulae)
    out = np.full(n, _LOG_FLOOR, dtype=np.float64)
    if n == 0:
        return out

    # Bucket by composition class.
    # Formulae with no carbon are automatically given minimum floor value
    rows: dict[str, list[int]] = {cls: [] for cls in CLASSES}
    feats: dict[str, list[np.ndarray]] = {cls: [] for cls in CLASSES}
    for i, formula in enumerate(formulae):
        feat = featurize_formula(formula)
        if feat is None:
            # i.e. carbon-less formula
            continue

        cls = _select_gmm_class(feat)
        rows[cls].append(i)
        feats[cls].append(feat)

    for cls in CLASSES:
        if not feats[cls]:
            continue
        gmm, tau, scale = gmm_params[cls]

        raw = gmm.score_samples(
            np.vstack(feats[cls])
        )

        out[rows[cls]] = _soft_gate_array(
            raw,
            tau,
            strength,
            softness * scale,
        )

    return out

def _soft_gate(
        raw: float,
        tau: float,
        strength: float,
        softness: float,
) -> float:
    """
    Smooth one-sided plausibility penalty,
     anchored at the floor `tau`

    Returns ~0 for a plausible formula (raw >> tau) and a negative penalty that
    grows with implausibility (raw << tau), floored at `_LOG_FLOOR`:

        penalty = -strength * softness * softplus(-(raw - tau) / softness)

    which is a softplus ramp of width ~`softness` (in nats of log-density) around
    tau with asymptotic slope `strength`.

    As `softness -> 0` this collapses to the hard hinge
    `strength * min(raw - tau, 0)`.

    Because this is applied at scoring time, the strength and softness
     params can be tuned without retraining GMM
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

def _soft_gate_array(
        raw: np.ndarray,
        tau: float,
        strength: float,
        softness: float,
) -> np.ndarray:
    """
    Vectorized version of `_soft_gate()`
     Must stay numerically identical to it -- `tests/test_prior.py`
     pins the two against each other.
    """
    d = np.asarray(raw, dtype=np.float64) - tau
    if softness <= 0.0:
        penalty = strength * np.minimum(d, 0.0)
    else:
        x = -d / softness
        softplus = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
        penalty = -strength * softness * softplus
    return np.maximum(penalty, _LOG_FLOOR)

#### TRAINING ####

def fit(
    formulae: list[str],
    n_components: int = 25,
    random_state: int = 42,
    tau_percentile: float = 0.5,
) -> GmmParams:
    """
    Trains two GMMs on a corpus of molecular formula strings.

    The corpus is split into composition classes (halogen vs halofree).
    Each class gets:
     - a GMM
     - a `tau`; i.e. the `tau_percentile` of training log-densities
            used by `log_prior` to designate implausible formulae
     - a `scale`; used as unit for 'softness'

    Raises ValueError if either classes has too few formulae
     to fit (< max(n_components, 10))

    Caches the fitted models on disk keyed by
    (corpus hash, n_components, random_state, tau_percentile)

    Args:
        formulae: List of formula strings (e.g. ["C6H12O6", ...])
        n_components: Number of Gaussian components per class
            (use select_n_components() to choose via BIC if unsure)
        random_state: Random seed
        tau_percentile: Percentile of each class's training log-densities used
                        as its plausibility floor. Lower = gentler gate.

    Returns:
        dict with keys 'halofree' and 'halogen',
        each storing a (GMM, tau, scale) tuple
    """
    cache_dir = _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    gmm_cache = (
            cache_dir
            / f'gmm_v{_CACHE_SCHEMA_VERSION}_{corpus_hash(formulae)}_k{n_components}'
              f'_rs{random_state}_p{tau_percentile}.json'
    )

    if gmm_cache.exists():
        return load(gmm_cache)

    # Parse corpus and construct ftrs matrix
    features = parse_corpus(formulae)

    # Train GMM for each class
    _halofree_ftrs, _halogen_ftrs = split_matrix_by_halogen(features)
    split = {
        'halofree': _halofree_ftrs,
        'halogen': _halogen_ftrs,
    }
    fitted_params: GmmParams = {}
    for cls in CLASSES:

        # 1. Fit GMM
        x = split[cls]
        if x.shape[0] < max(n_components, 10):
            raise ValueError(
                f"Too few {cls} examples in corpus"
            )

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state,
            n_init=3,
        )
        gmm.fit(x)
        train_scores = gmm.score_samples(x)

        # 2. Calculate tau (density at tau_percentile)
        tau = float(
            np.percentile(train_scores, tau_percentile)
        )

        # 3. Calculate scale (for softness)
        scale = float(np.std(train_scores))

        fitted_params[cls] = (
            gmm, tau, scale
        )

    # Save params in case the same GMM set is requested again
    save(
        path=gmm_cache,
        params=fitted_params,
    )

    return fitted_params


def parse_corpus(
        formulae: list[str]
) -> np.ndarray:
    """
    Parse formula strings into feature matrix, with disk caching
    (i.e. skips parsing if already cached)
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f'features_{corpus_hash(formulae)}.npy'

    if cache_path.exists():
        return np.load(cache_path)

    # Iterate over file and featurize
    rows = []
    for formula_str in formulae:
        formula_str = formula_str.strip()
        if not formula_str:
            continue
        try:
            f = Formula(formula_str)
        except Exception:
            continue

        feat = featurize_formula(f)
        if feat is not None:
            rows.append(feat)

    if len(rows) < 10:
        raise ValueError(
            f"Only {len(rows)} valid formulae parsed; need more data to train GMM"
        )

    features = np.array(rows)
    np.save(cache_path, features)
    return features


def select_n_components(
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

    features = parse_corpus(formulae)
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


def save(
        path: Path | str,
        params: GmmParams,
) -> None:
    """
    Save fitted per-class GMM parameters to a JSON file

    Layout: one block per composition class:
        {
            "halofree": {...},
            "halogen": {...}
        }
    """
    path = Path(path)
    data: dict[str, dict | None] = {}
    for cls in CLASSES:
        gmm, tau, scale = params[cls]
        data[cls] = {
            'n_components': gmm.n_components,
            'weights': gmm.weights_.tolist(),
            'means': gmm.means_.tolist(),
            'covariances': gmm.covariances_.tolist(),
            'tau': tau,
            'scale': scale,
        }

    path.write_text(
        json.dumps(data)
    )


def load(
        path: Path | str
) -> GmmParams:
    """
    Load per-class GMM parameters from a JSON file,
    skipping fitting

    Returns dict {
        'halofree': (gmm, tau, scale),
        'halogen': (gmm, tau, scale),
    }
    """
    data = json.loads(Path(path).read_text())
    try:
        return from_dict(data)
    except ValueError as e:
        raise ValueError(f"{e} (loading {path})") from e


def from_dict(
        data: dict,
) -> GmmParams:
    """
    Reconstruct per-class GMM parameters from a plain dict, skipping fitting.

    `data` has the same shape `save()` writes (and that
    `_coconut_gmm.DEFAULT_GMM_PARAMS` is pre-serialized in):
        {
            "halofree": {"n_components", "weights", "means", "covariances",
                         "tau", "scale"},
            "halogen": {...},
        }

    Returns dict {
        'halofree': (gmm, tau, scale),
        'halogen': (gmm, tau, scale),
    }
    """
    loaded: dict[str, tuple[GaussianMixture, float, float]] = {}
    for cls in CLASSES:
        block = data.get(cls)
        if block is None:
            raise ValueError(
                f"Missing key '{cls}' in GMM params"
            )

        # Initialize GMM then inject fitted parameters directly
        gmm = GaussianMixture(
            n_components=block['n_components'],
            covariance_type='full',
        )

        gmm.weights_ = np.array(block['weights'])
        gmm.means_ = np.array(block['means'])
        gmm.covariances_ = np.array(block['covariances'])
        gmm.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(gmm.covariances_)
        )

        loaded[cls] = (gmm, block['tau'], block['scale'])

    return loaded


def corpus_hash(
        formulae: list[str]
) -> str:
    """
    Return a hash of the corpus content
    (Used to check whether a GMM has already
    been trained on disk)
    """
    h = hashlib.sha256()
    for s in formulae:
        h.update(s.encode())
    return h.hexdigest()[:16]


def soft_gate(
        raw: float,
        tau: float,
        strength: float,
        softness: float,
) -> float:
    """
    Smooth one-sided plausibility penalty,
     anchored at the floor `tau`

    Returns ~0 for a plausible formula (raw >> tau) and a negative penalty that
    grows with implausibility (raw << tau), floored at `_LOG_FLOOR`:

        penalty = -strength * softness * softplus(-(raw - tau) / softness)

    which is a softplus ramp of width ~`softness` (in nats of log-density) around
    tau with asymptotic slope `strength`.

    As `softness -> 0` this collapses to the hard hinge
    `strength * min(raw - tau, 0)`.

    Because this is applied at scoring time, the strength and softness
     params can be tuned without retraining GMM
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


def soft_gate_array(
        raw: np.ndarray,
        tau: float,
        strength: float,
        softness: float,
) -> np.ndarray:
    """
    Vectorized version of `soft_gate()`
     Must stay numerically identical to it -- `tests/test_prior.py`
     pins the two against each other.
    """
    d = np.asarray(raw, dtype=np.float64) - tau
    if softness <= 0.0:
        penalty = strength * np.minimum(d, 0.0)
    else:
        x = -d / softness
        softplus = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
        penalty = -strength * softness * softplus
    return np.maximum(penalty, _LOG_FLOOR)

#### UTILS / HELPERS ####
def _select_gmm_class(
        feature: np.ndarray,
) -> Literal['halogen', 'halofree']:
    """
    Route a featurized formula to its composition-class model
    ('halogen' or 'halofree')
    """
    return 'halogen' if contains_halogen(feature) else 'halofree'