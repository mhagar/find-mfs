"""
Bayesian formula prior using element-ratio KDE distributions derived from
some formula database (i.e. NPAtlas)

This is a *prototype*

Models P(formula) using 1D KDEs on X/C ratios (for X in {H, N, O, P, S}).

Note - because for most elements, X=0, each ratio uses a
    zero-inflated mixture:

    i.e. if a formula doesn't have a sulfur, then P(formula) is
        the fraction of formulae in corpus that don't have sulfurs.

    if a formula DOES have a sulfur, *THEN* use KDE

    P(ratio) = P(absent)                if element count is 0
               P(present) * kde(ratio)  if element count > 0


log P(formula) = sum of log P(ratio_i) across all ratio elements.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from molmass import Formula
from scipy.stats import gaussian_kde

from ..core.light_formula import LightFormula

if TYPE_CHECKING:
    from ..core.finder import FormulaCandidate
    from ..core.results import FormulaSearchResults

# Elements to model X/C for
_RATIO_ELEMENTS = ('H', 'N', 'O', 'S', 'P')

# Small uniform weight to prevent -inf scores
_UNIFORM_WEIGHT = 1e-6


class FormulaPrior:
    """
    Corpus-derived prior P(formula) based on element ratio distributions. Used
    to assess 'formula plausibility'

    Learns what "normal" element compositions look like from a list of
    known formulae (e.g. from a metabolite database), then scores new
    candidates accordingly.

    Example:
        >>> corpus = ["C6H12O6", "C12H22O11", "C27H46O", "C5H9NO4"]
        >>> prior = FormulaPrior().fit(corpus)
        >>> prior.log_prior(Formula("C6H12O6"))
        >>> # -0.125
    """

    def __init__(self):
        self._kdes: dict[str, gaussian_kde] = {}
        self._p_absent: dict[str, float] = {}
        self._fitted = False

    def fit(
        self,
        formulae: list[str],
        plot_dir: Path | str | None = None,
    ) -> 'FormulaPrior':
        """
        Learn element ratio distributions from a corpus of formula strings.

        Args:
            formulae: List of molecular formula strings (e.g. ["C6H12O6", ...])
            plot_dir: If given, save one KDE plot per element as a PNG in this directory.

        Returns:
            self, for chaining
        """
        # Parse formulae and collect ratios
        ratios: dict[str, list[float]] = {
            elem: [] for elem in _RATIO_ELEMENTS
        }

        for formula_str in formulae:
            f = Formula(formula_str)
            comp = f.composition()

            c_count = comp['C'].count if 'C' in comp else 0
            if c_count == 0:
                continue  # skip formulae without carbon

            for elem in _RATIO_ELEMENTS:
                count = comp[elem].count if elem in comp else 0
                ratios[elem].append(count / c_count)

        # Fit KDE for each element ratio
        for elem in _RATIO_ELEMENTS:
            values = ratios[elem]
            if not values:
                self._p_absent[elem] = 1.0
                continue

            n_total = len(values)
            nonzero = [v for v in values if v > 0]
            n_absent = n_total - len(nonzero)

            self._p_absent[elem] = n_absent / n_total

            if len(nonzero) >= 2:
                self._kdes[elem] = gaussian_kde(nonzero)
            elif len(nonzero) == 1:
                # Single data point: use a narrow kde by adding slight jitter
                self._kdes[elem] = gaussian_kde(
                    [nonzero[0], nonzero[0] * 1.01]
                )

        self._fitted = True

        if plot_dir is not None:
            self._save_kde_plots(Path(plot_dir), ratios)

        return self

    def _save_kde_plots(
        self,
        plot_dir: Path,
        ratios: dict[str, list[float]],
    ) -> None:
        import matplotlib.pyplot as plt

        plot_dir.mkdir(parents=True, exist_ok=True)

        for elem in _RATIO_ELEMENTS:
            kde = self._kdes.get(elem)
            values = ratios.get(elem, [])
            nonzero = [v for v in values if v > 0]

            fig, ax = plt.subplots(figsize=(6, 4))

            if kde is not None and nonzero:
                x = np.linspace(0, max(nonzero) * 1.2, 500)
                ax.hist(nonzero, bins=40, density=True, alpha=0.4,
                        color='steelblue', label='data')
                ax.plot(x, kde.evaluate(x), color='steelblue', lw=2,
                        label='KDE')

            p_absent = self._p_absent.get(elem, 1.0)
            ax.set_title(
                f'{elem}/C ratio  |  '
                f'absent: {p_absent:.1%}  present: {1 - p_absent:.1%}'
            )
            ax.set_xlabel(f'{elem}/C')
            ax.set_ylabel('density')
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f'kde_{elem}_C.png', dpi=150)
            plt.close(fig)

    def log_prior(
        self,
        formula: Formula | LightFormula,
    ) -> float:
        """
        Compute log P(formula) under the fitted prior

        Args:
            formula: A molmass.Formula or LightFormula instance

        Returns:
            Log-probability score; higher = more plausible.
            Returns 0.0 (uninformative) for formulae without carbon.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before log_prior()")

        # Extract element counts via duck-typed composition()
        elem_counts = _get_element_counts(formula)

        c_count = elem_counts.get('C', 0)
        if c_count == 0:
            return 0.0  # uninformative for non-carbon formulae

        log_p = 0.0

        for elem in _RATIO_ELEMENTS:
            count = elem_counts.get(elem, 0)
            p_absent = self._p_absent.get(elem, 1.0)
            p_present = 1.0 - p_absent

            if count == 0:
                # Element is absent
                # Just use the fraction of elements in corpus without element
                log_p += math.log(p_absent + _UNIFORM_WEIGHT)

            else:
                # Element is present
                # Use KDE
                ratio = count / c_count
                kde = self._kdes.get(elem)
                if kde is not None:
                    kde_val = float(kde.evaluate(np.array([ratio]))[0])
                    log_p += math.log(
                        p_present * kde_val + _UNIFORM_WEIGHT
                    )
                else:
                    # No KDE fitted (element never seen in corpus)
                    log_p += math.log(_UNIFORM_WEIGHT)

        return log_p

    def score_results(
        self,
        results: 'FormulaSearchResults',
        mass_sigma_ppm: float,
        isotope_sigma: float,
    ) -> None:
        """
        Score all candidates using the full posterior and return sorted results.

        Computes:
            log P(formula | data) = log P(prior)
                                   - Δm² / (2 * mass_sigma_ppm²)
                                   - RMSE² / (2 * isotope_sigma²)  [if available]

        Args:
            results: FormulaSearchResults to score
            mass_sigma_ppm: Instrument mass accuracy in ppm
            isotope_sigma: Expected isotope intensity error

        Returns:
            None (results are scored in-place)
        """

        for candidate in results.candidates:
            candidate: 'FormulaCandidate'
            log_posterior = self.log_prior(candidate.formula)
            candidate.prior_score = log_posterior

            # Mass error likelihood term
            log_posterior -= (
                ( candidate.error_ppm ** 2) /
                (2 * mass_sigma_ppm ** 2)
            )

            # Isotope likelihood term (if available)
            if candidate.isotope_match_result is not None:
                rmse = candidate.isotope_match_result.intensity_rmse
                log_posterior -= (
                    (rmse ** 2) /
                    (2 * isotope_sigma ** 2)
                )

            candidate.posterior_score = log_posterior



def _get_element_counts(formula) -> dict[str, int]:
    """
    Extract element counts from a Formula or LightFormula
    """
    counts: dict[str, int] = {}
    # Works for both Formula and LightFormula via composition()
    comp = formula.composition()
    for symbol, item in comp.items():
        if symbol == '' or symbol == 'e-':
            continue
        if item.count > 0:
            counts[symbol] = item.count
    return counts
