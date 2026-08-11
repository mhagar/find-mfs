"""
This module has the FormulaSearchResults class, which contains
FormulaCandidate objects, and provides convenience methods for:
- filtering,
- display
- export
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, overload, TYPE_CHECKING

import molmass
import numpy as np

from .finder import FormulaCandidate
from .light_formula import LightFormula
from ..utils.filtering import passes_octet_rule
from ..utils.table import render_table, render_dataframe

if TYPE_CHECKING:
    import pandas as pd


class _LazyBackend:
    """
    Stores raw numpy arrays and materializes FormulaCandidate on demand.

    This avoids eagerly constructing N LightFormula + N FormulaCandidate
    objects when the user may only inspect a few of them.
    """
    __slots__ = (
        '_counts', '_exact_masses', '_error_ppm', '_error_da',
        '_rdbe',
        '_symbols', '_charge', '_ion_charge', '_adduct', '_adduct_elements',
        '_charge_mass_offset', '_adduct_mass',
        '_cache',
    )

    def __init__(
        self,
        raw: dict,
        symbols: list[str],
        charge: int,
        ion_charge: int,
        adduct: str | None = None,
        adduct_elements: dict[str, int] | None = None,
        charge_mass_offset: float = 0.0,
        adduct_mass: float = 0.0,
    ):
        self._counts = raw['counts']
        self._exact_masses = raw['exact_masses']
        self._error_ppm = raw['error_ppm']
        self._error_da = raw['error_da']
        self._rdbe = raw.get('rdbe')
        self._symbols = symbols
        self._charge = charge
        self._ion_charge = ion_charge
        self._adduct = adduct
        self._adduct_elements = adduct_elements
        self._charge_mass_offset = charge_mass_offset
        self._adduct_mass = adduct_mass
        self._cache: dict[int, FormulaCandidate] = {}

    def __len__(self) -> int:
        return self._counts.shape[0]

    def _build_ion_formula(
        self,
        idx: int,
        row_list: list[int],
        core_formula: LightFormula,
    ) -> LightFormula | None:
        if self._adduct_elements is None:
            return core_formula

        ion_elements = {
            sym: count for sym, count in zip(self._symbols, row_list) if count > 0
        }
        for sym, delta in self._adduct_elements.items():
            updated = ion_elements.get(sym, 0) + delta
            if updated < 0:
                return None
            if updated == 0:
                ion_elements.pop(sym, None)
            else:
                ion_elements[sym] = updated

        return LightFormula(
            elements=ion_elements,
            charge=self._ion_charge,
            monoisotopic_mass=float(self._exact_masses[idx]),
        )

    def _materialize(self, idx: int) -> FormulaCandidate:
        if idx in self._cache:
            return self._cache[idx]

        row_list = self._counts[idx].tolist()

        if self._adduct is not None:
            # Adduct path: core molecule is neutral
            formula = LightFormula.from_counts(
                symbols=self._symbols,
                counts=row_list,
                charge=0,
                monoisotopic_mass=(
                    float(self._exact_masses[idx])
                    + self._charge_mass_offset
                    - self._adduct_mass
                ),
            )
        else:
            formula = LightFormula.from_counts(
                symbols=self._symbols,
                counts=row_list,
                charge=self._charge,
                monoisotopic_mass=float(self._exact_masses[idx]),
            )

        # Ion formula (core + signed adduct offsets), used for isotope-envelope
        # simulation during scoring. None if the adduct removes more atoms than
        # the candidate has (chemically invalid ion) — such candidates remain in
        # the results but cannot be isotope-scored.
        ion_formula = self._build_ion_formula(
            idx=idx,
            row_list=row_list,
            core_formula=formula,
        )

        candidate = FormulaCandidate(
            formula=formula,
            error_ppm=float(self._error_ppm[idx]),
            error_da=float(self._error_da[idx]),
            rdbe=float(self._rdbe[idx]) if self._rdbe is not None else None,
            adduct=self._adduct,
            ion_formula=ion_formula,
        )
        self._cache[idx] = candidate
        return candidate

    def _reindex(self, idx) -> '_LazyBackend':
        """
        Return a new _LazyBackend reindexed
        by slice, boolean mask, or int array
        """
        raw = {
            'counts': self._counts[idx],
            'exact_masses': self._exact_masses[idx],
            'error_ppm': self._error_ppm[idx],
            'error_da': self._error_da[idx],
        }
        if self._rdbe is not None:
            raw['rdbe'] = self._rdbe[idx]
        return _LazyBackend(
            raw=raw,
            symbols=self._symbols,
            charge=self._charge,
            ion_charge=self._ion_charge,
            adduct=self._adduct,
            adduct_elements=self._adduct_elements,
            charge_mass_offset=self._charge_mass_offset,
            adduct_mass=self._adduct_mass,
        )

    def _slice(self, s: slice) -> '_LazyBackend':
        """
        Return a new _LazyBackend for a slice of the data
        """
        return self._reindex(s)

    def _filter_by_mask(self, mask: np.ndarray) -> '_LazyBackend':
        """
        Return a new _LazyBackend filtered by boolean mask
        """
        return self._reindex(mask)


@dataclass
class FormulaSearchResults:
    """
    Container for formula search results with filtering and display methods

    This class wraps a list of FormulaCandidate objects and provides:
    - Iterator/indexing support for easy access to MF candidates
    - Post-hoc filtering methods that return new FormulaSearchResults
    - Formatted representation in response to `print()`
    - Formatted table output via to_table()
    - Optional pandas DataFrame export

    Attributes:
        candidates: List of formula candidates
        query_mass: The mass that was searched
        query_params: Dictionary of search parameters used

    Example:
        >>> finder: 'FormulaFinder'
        >>> results = finder.find_formulae(mass=180.063, error_ppm=5.0)
        >>> print(results)  # Gives a summary
        >>> for candidate in results:  # Iterate
        ...     print(candidate.formula)
        >>> # Post-hoc filter:
        >>> filtered: FormulaSearchResults = results.filter_by_rdbe(0, 10)
    """
    candidates: list[FormulaCandidate]
    query_mass: float
    query_params: dict = field(default_factory=dict)
    _backend: _LazyBackend | None = field(default=None, repr=False)

    def __getattribute__(self, name):
        if name == 'candidates':
            backend = object.__getattribute__(self, '_backend')
            if backend is not None:
                return [
                    backend._materialize(i) for i in range(len(backend))
                ]
        return super().__getattribute__(name)

    def __len__(self) -> int:
        if self._backend is not None:
            return len(self._backend)
        return len(self.candidates)

    def __iter__(self):
        if self._backend is not None:
            return (self._backend._materialize(i) for i in range(len(self._backend)))
        return iter(self.candidates)

    @overload
    def __getitem__(
        self,
        idx: int,
    ) -> FormulaCandidate: ...

    @overload
    def __getitem__(
        self,
        idx: slice
    ) -> 'FormulaSearchResults': ...

    def __getitem__(
        self,
        idx: int | slice,
    ) -> 'FormulaCandidate | FormulaSearchResults':
        """
        Either returns a formula candidate, or a new
        FormulaSearchResults instance if given a slice

        Args:
            idx: index or slice of list

        Returns:
            Either a FormulaCandidate object, or
            another FormulaSearchResults instance
        """
        if self._backend is not None:
            if isinstance(idx, slice):
                return FormulaSearchResults(
                    candidates=[],
                    query_mass=self.query_mass,
                    query_params=self.query_params,
                    _backend=self._backend._slice(idx),
                )
            if idx < 0:
                idx += len(self._backend)
            return self._backend._materialize(idx)

        if isinstance(idx, slice):
            return FormulaSearchResults(
                candidates=self.candidates[idx],
                query_mass=self.query_mass,
                query_params=self.query_params,
            )

        return self.candidates[idx]

    def __repr__(self) -> str:
        """
        Text summary and top candidates
        """
        n_results = len(self.candidates)
        summary = self._summary_line(n_results)

        if n_results == 0:
            return summary

        # Show top 5 candidates
        lines = [summary, "", self.to_table(max_rows=5)]
        return "\n".join(lines)

    # === FORMATTING METHODS ===
    def _summary_line(self, n_results: int) -> str:
        """
        Build the header line, including adduct notation when present.
        """
        adduct = self.query_params.get('adduct')
        charge = self.query_params.get('charge', 0)
        parts = [
            f"query_mass={self.query_mass:.4f}",
            f"n_results={n_results}",
        ]

        if adduct is not None:
            adduct_part = adduct if adduct.startswith('-') else f'+{adduct}'
            sign = '+' if charge > 0 else '-' if charge < 0 else ''
            abs_charge = abs(charge)
            charge_str = f'{abs_charge}{sign}' if abs_charge > 1 else sign
            parts.append(f"adduct=[M{adduct_part}]{charge_str}")

        return f"FormulaSearchResults({', '.join(parts)})"

    def to_table(
        self,
        max_rows: Optional[int] = None
    ) -> str:
        """
        Return formatted table of all candidates

        Args:
            max_rows: Maximum number of rows to display. None shows all.

        Returns:
            Formatted string table
        """
        candidates_to_show = self.candidates[:max_rows] \
            if max_rows is not None else self.candidates

        return render_table(
            candidates_to_show,
            max_rows=max_rows,
            total=len(self.candidates)
        )

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert results to pandas DataFrame, if pandas is installed.

        Columns match those shown in to_table(), with conditional score columns
        (chem_logprior, iso_loglik, mass_loglik, ms2_loglik, log_posterior)
        included only
        when present.

        Returns:
            pandas.DataFrame with columns for formula, errors, RDBE, and
            any scored columns that were computed

        Raises:
            ImportError: If pandas is not installed
        """
        return render_dataframe(self.candidates)

    # === SORTING METHODS ===
    def sort_by_error(
        self,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by absolute mass error (Da).

        Args:
            reverse: If True, sort in descending order (largest error first)

        Returns:
            New FormulaSearchResults with sorted candidates
        """
        # if self._backend is not None:
        #     b = self._backend
        #     order = np.argsort(np.abs(b._error_da))
        #     if reverse:
        #         order = order[::-1]
        #     new_backend = b._reindex(order)
        #     return FormulaSearchResults(
        #         candidates=[], query_mass=self.query_mass,
        #         query_params=self.query_params, _backend=new_backend,
        #     )

        return FormulaSearchResults(
            candidates=sorted(self.candidates, reverse=reverse),
            query_mass=self.query_mass,
            query_params=self.query_params,
        )

    def _sort_by_score(
        self,
        field: str,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by a scalar log-score attribute (descending by default,
        so the most plausible candidate is first). Candidates whose score is
        None (never scored) are placed at the end.

        Args:
            field: FormulaCandidate attribute name holding the score.
            reverse: If True, sort ascending (lowest score first) instead.
        """
        with_score = [c for c in self.candidates if getattr(c, field) is not None]
        without_score = [c for c in self.candidates if getattr(c, field) is None]

        sorted_with = sorted(
            with_score,
            key=lambda x: getattr(x, field),
            reverse=not reverse,
        )

        return FormulaSearchResults(
            candidates=sorted_with + without_score,
            query_mass=self.query_mass,
            query_params=self.query_params,
        )

    def sort_by_iso_loglik(
        self,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by isotope log-likelihood (descending by
        default).

        :param reverse:
        :return:
        """
        return self._sort_by_score('iso_loglik', reverse=reverse)

    def sort_by_chem_logprior(
        self,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by chemical-plausibility log-prior (descending by
        default). Candidates without a score are placed at the end.

        Args:
            reverse: If True, sort ascending (lowest score first) instead.
        """
        return self._sort_by_score('chem_logprior', reverse=reverse)

    def sort_by_ms2_logit(
        self,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """Sort candidates by raw MS2 reranker logit (best first by default)."""
        return self._sort_by_score('ms2_logit', reverse=reverse)

    # === SET-DEPENDENT SCORES ===
    #
    # These are computed on demand, never stored. The MS2 term is a softmax over
    # the candidate set, i.e. is not a candidate-by-candidate value
    def ms2_loglik(
        self,
        temperature: float = 1.0,
        unscored: str = 'floor',
    ) -> np.ndarray:
        """
        log P(formula | MS2) for the current candidate set

        The reranker is trained with a softmax-over-candidates objective, so
        normalizing its logits across the candidate set is what turns them into
        a probability. That makes the value set-dependent: filter the results
        and every entry changes. This recomputes, so it is always consistent
        with `self.candidates`.

        Args:
            temperature: Softmax temperature, applied to the logits before
                normalizing. T < 1 sharpens the MS2 term (bigger gaps between
                candidates, MS2 dominates the other posterior terms); T > 1
                flattens it into a weaker vote. T = 1 is the reranker's own
                calibrated distribution, since it was trained with a
                softmax-over-candidates objective.
            unscored: What to give candidates with no `ms2_logit` (e.g. ones
                skipped by a top-N cascade).
                'floor' assigns the minimum log-likelihood among scored
                candidates, so an unscored candidate can never outrank a scored
                one on MS2 evidence alone.
                'nan' marks them instead, for callers that want to handle the
                gap themselves.

        Returns:
            Array aligned to `self.candidates`. All zeros if nothing was
            MS2-scored, i.e. the term is simply absent.

        Raises:
            ValueError: If `temperature` is not positive, or `unscored` is not
                a recognised policy.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if unscored not in ('floor', 'nan'):
            raise ValueError(f"unscored must be 'floor' or 'nan', got {unscored!r}")

        logits = np.array(
            [
                np.nan if c.ms2_logit is None else c.ms2_logit
                for c in self.candidates
            ],
            dtype=np.float64,
        )
        scored = ~np.isnan(logits)
        out = np.zeros(len(logits), dtype=np.float64)
        if not scored.any():
            return out

        shifted = logits[scored] / temperature
        shifted -= shifted.max()
        out[scored] = shifted - np.log(np.exp(shifted).sum())
        out[~scored] = out[scored].min() if unscored == 'floor' else np.nan
        return out

    def log_posterior(
        self,
        ms2_weight: float = 1.0,
        ms2_temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Full stacked log-posterior for the current candidate set.

        Adds the set-normalized MS2 term to each candidate's stored
        per-candidate terms:

            candidate.log_posterior + ms2_weight * ms2_loglik()

        Note:
            `ms2_weight` and `ms2_temperature` are **not independent dials for
            ranking purposes** -- only their ratio matters. Expanding the MS2
            term for candidate `i` with logits `z`::

                w * (z_i / T - logsumexp(z / T))  ==  (w / T) * z_i - w * C

            `C` is identical for every candidate in the set, so it cannot change
            the ordering. `ms2_weight=1, ms2_temperature=0.5` therefore ranks
            exactly like `ms2_weight=2, ms2_temperature=1`.

            They diverge only in the *absolute* value of the posterior, which
            matters if you compare scores across spectra or threshold on them --
            and `ms2_temperature=1` is the one setting where the term is a
            genuine calibrated log-probability. In practice: use `ms2_weight=0`
            to switch MS2 off, and tune one of the two, not both.

            (The equivalence is exact over scored candidates; the
            `unscored='floor'` policy is not linear in `z`, so candidates
            skipped by a cascade can shift slightly.)

        Args:
            ms2_weight: Weight on the MS2 term. 0 reproduces the per-candidate
                `candidate.log_posterior` exactly.
            ms2_temperature: Passed to :meth:`ms2_loglik`.

        Returns:
            Array aligned to `self.candidates`. Unscored candidates contribute
            0 for any term that is None.
        """
        base = np.array(
            [
                0.0 if c.log_posterior is None else c.log_posterior
                for c in self.candidates
            ],
            dtype=np.float64,
        )
        if not ms2_weight:
            return base
        return base + ms2_weight * self.ms2_loglik(temperature=ms2_temperature)

    def sort_by_posterior(
        self,
        reverse: bool = False,
        ms2_weight: float = 1.0,
        ms2_temperature: float = 1.0,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by the full stacked log-posterior (descending by
        default, so the top-ranked candidate is first). Candidates that were
        never scored are placed at the end.

        The MS2 term is recomputed over the current candidate set rather than
        read from a stored value (see :meth:`ms2_loglik`), so sorting a filtered
        result ranks by a posterior that is correct for what actually remains.

        Args:
            reverse: If True, sort ascending (lowest score first) instead.
            ms2_weight: Weight on the MS2 term. 0 sorts by the per-candidate
                terms alone.
            ms2_temperature: Passed to :meth:`ms2_loglik`. Note that only
                `ms2_weight / ms2_temperature` affects the ordering -- see
                :meth:`log_posterior`.
        """
        candidates = self.candidates
        posterior = self.log_posterior(
            ms2_weight=ms2_weight, ms2_temperature=ms2_temperature
        )

        scored = [
            (p, c) for p, c in zip(posterior, candidates)
            if c.log_posterior is not None or c.ms2_logit is not None
        ]
        unscored = [
            c for c in candidates
            if c.log_posterior is None and c.ms2_logit is None
        ]
        scored.sort(key=lambda pc: pc[0], reverse=not reverse)

        return FormulaSearchResults(
            candidates=[c for _, c in scored] + unscored,
            query_mass=self.query_mass,
            query_params=self.query_params,
        )

    # === FILTERING METHODS ===
    def filter_by_rdbe(
        self,
        min_rdbe: float,
        max_rdbe: float
    ) -> 'FormulaSearchResults':
        """
        Filter candidates by RDBE range

        Args:
            min_rdbe: Minimum RDBE value (inclusive)
            max_rdbe: Maximum RDBE value (inclusive)

        Returns:
            New FormulaSearchResults with filtered candidates
        """
        if self._backend is not None:
            b = self._backend
            if b._rdbe is None:
                # No RDBE data — cannot filter, return empty
                return FormulaSearchResults(
                    candidates=[], query_mass=self.query_mass,
                    query_params={**self.query_params, 'filter_rdbe': (min_rdbe, max_rdbe)},
                )
            mask = (b._rdbe >= min_rdbe) & (b._rdbe <= max_rdbe)
            new_backend = b._filter_by_mask(mask)
            return FormulaSearchResults(
                candidates=[], query_mass=self.query_mass,
                query_params={**self.query_params, 'filter_rdbe': (min_rdbe, max_rdbe)},
                _backend=new_backend,
            )

        filtered = [
            c for c in self.candidates
            if c.rdbe is not None and min_rdbe <= c.rdbe <= max_rdbe
        ]

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'filter_rdbe': (min_rdbe, max_rdbe),
            }
        )

    def filter_by_octet(self) -> 'FormulaSearchResults':
        """
        Filter candidates to only those passing the octet rule.

        Returns:
            New FormulaSearchResults with filtered candidates
        """
        # Octet filtering requires materializing formulas
        filtered = [
            c for c in self
            if passes_octet_rule(c.formula)
        ]

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'check_octet': True,
            }
        )

    def filter_by_error(
        self,
        max_ppm: Optional[float] = None,
        max_da: Optional[float] = None
    ) -> 'FormulaSearchResults':
        """
        Filter candidates by maximum error.

        At least one of max_ppm or max_da must be specified.

        Args:
            max_ppm: Maximum absolute error in ppm
            max_da: Maximum absolute error in Da

        Returns:
            New FormulaSearchResults with filtered candidates

        Raises:
            ValueError: If neither max_ppm nor max_da is specified
        """
        if max_ppm is None and max_da is None:
            raise ValueError(
                "At least one of max_ppm or max_da must be specified"
            )

        if self._backend is not None:
            b = self._backend
            mask = np.ones(len(b), dtype=bool)
            if max_ppm is not None:
                mask &= np.abs(b._error_ppm) <= max_ppm
            if max_da is not None:
                mask &= np.abs(b._error_da) <= max_da
            new_backend = b._filter_by_mask(mask)
            return FormulaSearchResults(
                candidates=[], query_mass=self.query_mass,
                query_params={
                    **self.query_params,
                    'max_error_ppm': max_ppm, 'max_error_da': max_da,
                },
                _backend=new_backend,
            )

        filtered = []
        for c in self.candidates:
            passes = True
            if max_ppm is not None and abs(c.error_ppm) > max_ppm:
                passes = False
            if max_da is not None and abs(c.error_da) > max_da:
                passes = False
            if passes:
                filtered.append(c)

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'max_error_ppm': max_ppm,
                'max_error_da': max_da
            }
        )

    def top(
        self,
        n: int = 10,
    ) -> 'FormulaSearchResults':
        """
        Return top N candidates by error.

        Args:
            n: Number of top candidates to return

        Returns:
            New FormulaSearchResults with top N candidates
        """
        return self[:n]

    def contains_formula(
        self,
        query: str | LightFormula | molmass.Formula
    ) -> Optional[FormulaCandidate]:
        """
        If the search results contain the given query formula,
        returns the entry
        """
        if isinstance(query, str):
            query = molmass.Formula(query)

        for candidate in self.candidates:
            if candidate.formula.formula == query.formula:
                return candidate

        return None


