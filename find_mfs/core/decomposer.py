"""
Mass decomposition using Bocker & Liptak algorithm,
(as implemented in SIRIUS) i.e. using an extended residue table
"""
import math
from typing import Optional, Iterable, TYPE_CHECKING

import numpy as np
from molmass import Formula
from molmass.elements import ELEMENTS, ELECTRON
from numba import int32, float64, types
from numba.typed import List as NumbaList  # nb will deprecate list reflection
from numba.experimental import jitclass

from .algorithms import _gcd, _decompose_integer_mass

if TYPE_CHECKING:
    from molmass import Isotope, Element


# jitclass specification
spec = [
    ('symbol', types.unicode_type),  # For string
    ('mass', float64),
    ('integer_mass', int32),
]

@jitclass(spec)
class Element:
    """
    Numba-compatible dataclass representing a
    chemical element with its mass and symbol
    """
    def __init__(
        self,
        symbol: str,
        mass: float
    ):
        self.symbol = symbol
        self.mass = mass
        self.integer_mass = 0  # Will be set during discretization


class MassDecomposer:
    """
    Decomposes a mass into candidate molecular formulae

    Each MassDecomposer object is only viable for the elements
    dict it's initialized with. This is because upon initialization,
    an extended residue table (ERT) is constructed to enable
    rapid mass decomposition [1].

    If no `elements` argument is given, initializes with CHNOPS
    by default. This package actually comes with a pre-computed
    ERT for CHNOPS and CHNOPS + Halogens, so if either of those
    element sets are requested, initialization is much faster.

    Otherwise, the package will fall back to calculating a new
    ERT from scratch (though this only takes a few seconds).

    [1]:
    Bocker & Liptak:
    "A Fast and Simple Algorithm for the Money Changing Problem"
    doi: 10.1007/s00453-007-0162-8
    """

    def __init__(
        self,
        elements: Iterable[str] = 'CHNOPS',
        use_precalculated: bool = True,
    ):
        """
        Initializes with a dictionary of elements and their masses

        Args:

            elements:
                List of elements to be considered for mass decomposition.
                Default is ['C', 'H', 'N', 'O', 'P', 'S']

            use_precalculated:
                Whether to use pre-calculated ERTs when available.
                Default is True for faster initialization
        """
        # Convert elements into a list of symbols, if it's a string
        if isinstance(elements, str):
            elements = list(Formula(elements).composition().keys())

        # Try to load pre-calculated ERT if available + requested
        if use_precalculated and self._try_load_precalculated(elements):
            return  # Successfully loaded pre-calculated ERT

        # Fall back to standard ERT construction
        self._init_from_elements(elements)


    # *** INITIALIZING EXTENDED RESIDUE TABLE ***
    def init_ERT(self):
        """
        Build extended residue table, which is used to quickly
        figure out whether a mass is decomposable using the
        available elements

        Used to prune search tree
        """
        if self.ERT is not None:
            return

        self._discretize_masses()
        self._divide_by_gcd()
        self._calculate_ert()
        self._compute_errors()

    def _discretize_masses(self):
        """
        Convert floating-point masses to integers (faster handling)
        i.e. H becomes 6011 (instead of 1.007...)

        Mass relationships preserved because everything scaled by same factor
        """
        for element in self.elements:
            element.integer_mass = int(element.mass / self.precision)

    def _divide_by_gcd(self):
        """
        Divide all masses by their greatest common divisor to
        reduce computation later on (shrinking problem size)
        """
        # Single element case:
        if len(self.elements) == 1:
            d = self.elements[0].integer_mass
            self.precision *= d
            self.elements[0].integer_mass = 1

        elif len(self.elements) > 1:

            # Find GCD of all masses
            d = _gcd(self.elements[0].integer_mass, self.elements[1].integer_mass)
            for i in range(2, len(self.elements)):
                d = _gcd(d, self.elements[i].integer_mass)
                if d == 1:
                    return  # No further reduction possible

            # Scale precision and adjust integer masses
            self.precision *= d
            for element in self.elements:
                element.integer_mass //= d

    def _calculate_ert(self):
        """
        Compute the extended residue table using the proxy masses
        """
        first_mass = self.elements[0].integer_mass
        num_elements = len(self.elements)

        # Initialize ERT table
        self.ERT = np.full(
            shape=(first_mass, num_elements),
            fill_value=np.inf,
            dtype=np.float64,
        )
        self.ERT[0, 0] = 0  # Base case

        # Fill first column (only first element)
        for i in range(1, first_mass):
            if i % first_mass == 0:
                self.ERT[i, 0] = i

        # Fill remaining columns
        for j in range(1, num_elements):
            self.ERT[0, j] = 0  # Base case for each column
            current_mass = self.elements[j].integer_mass
            d = _gcd(first_mass, current_mass)

            # Round-robin loops for residue classes
            for p in range(d):
                if p == 0:
                    n = 0  # Start with 0 for first residue class
                else:
                    # Find minimum in specific part of ERT
                    n = np.inf
                    argmin = p
                    for i in range(p, first_mass, d):
                        if self.ERT[i, j-1] < n:
                            n = self.ERT[i, j-1]
                            argmin = i

                    if np.isinf(n):
                        # Fill specific part of ERT with infinity
                        for i in range(p, first_mass, d):
                            self.ERT[i, j] = np.inf
                        continue

                    self.ERT[argmin, j] = n

                # Normal loop to fill remaining cells
                for i in range(1, first_mass // d):
                    n += current_mass
                    r = int(n % first_mass)
                    n = min(n, self.ERT[r, j-1])
                    self.ERT[r, j] = n

    def _compute_errors(self):
        """
        Compute maximum relative errors due to mass discretization
        """
        self.min_error = 0
        self.max_error = 0

        for element in self.elements:
            error = (self.precision * element.integer_mass - element.mass) / element.mass
            self.min_error = min(self.min_error, error)
            self.max_error = max(self.max_error, error)

    def _try_load_precalculated(
        self,
        elements: Iterable[str]
    ) -> bool:
        """
        Try to load a pre-calculated ERT for the given element set.

        Returns:
            True if successfully loaded, False otherwise
        """
        elements: list[str] = list(elements)

        # Check for exact matches with known pre-calculated sets
        known_sets = {
            frozenset(['C', 'H', 'N', 'O', 'P', 'S']): 'chnops',
            frozenset(['C', 'H', 'N', 'O', 'P', 'S', 'Br', 'Cl', 'I', 'F']): 'chnops_halogens'
        }

        element_set = frozenset(elements)
        if element_set not in known_sets:
            # Triggers standard ERT construction
            return False

        ert_name = known_sets[element_set]

        try:
            # Import the pre-calculated ERT module
            if ert_name == 'chnops':
                from ..data.ert_chnops import ERT_DATA
            elif ert_name == 'chnops_halogens':
                from ..data.ert_chnops_halogens import ERT_DATA
            else:
                return False  # Unknown ERT

            self._load_from_ert_data(ERT_DATA)
            return True

        except (ImportError, KeyError, AttributeError):
            # Pre-calculated ERT module not found or corrupted,
            #   fall back to construction
            return False

    def _load_from_ert_data(
        self,
        ert_data: dict,
    ):
        """
        Load ERT and related data from a pre-calculated dictionary
        """
        self.ERT = ert_data['ert']
        self.precision = float(ert_data['precision'])
        self.min_error = float(ert_data['min_error'])
        self.max_error = float(ert_data['max_error'])
        self.element_symbols = list(ert_data['element_symbols'])

        # Reconstruct Element objects
        element_masses = ert_data['element_masses']
        element_integer_masses = ert_data['element_integer_masses']

        self.elements = NumbaList()
        for i, symbol in enumerate(self.element_symbols):
            element = Element(symbol, element_masses[i])
            element.integer_mass = element_integer_masses[i]
            self.elements.append(element)

    def _init_from_elements(self, elements):
        """
        Initialize ERT from scratch using the Bocker algorithm
        """
        # Create Numba-compatible Element objects
        self.elements = NumbaList()
        for symbol in elements:
            if symbol not in ELEMENTS:
                raise ValueError(
                    f"Invalid chemical symbol, not recognized as element:"
                    f" {symbol}"
                )

            mass = get_element_most_abundant_mass(
                element=ELEMENTS[symbol]
            )

            self.elements.append(
                Element(symbol, mass),
            )

        # Sort elements by mass (ascending)
        self.elements.sort(
            key=lambda e: e.mass
        )
        self.element_symbols = [i.symbol for i in self.elements]

        # Default precision factor used by SIRIUS.
        # I'm unsure how they settled on this value! :)
        self.precision = 1.0 / 5963.337687

        # ERT computed during initialization
        self.ERT: Optional[np.ndarray[int, int]] = None
        self.min_error = 0.0
        self.max_error = 0.0

        self.init_ERT()


    # *** MASS DECOMPOSITION USING ERT***
    def decompose(
        self,
        query_mass: float,
        charge: int = 0,
        ppm_error: Optional[float] = 0.0,
        mz_error: Optional[float] = 0.0,
        min_counts: Optional[dict[str, int]] = None,
        max_counts: Optional[dict[str, int]] = None,
        max_results: int = 10000,
    ) -> list[Formula]:
        """
        Decompose a mass into possible element combinations within some
        error window.

        This method performs pure mass decomposition using the Extended
        Residue Table algorithm. It does NOT apply any chemical validation
        or filtering - use FormulaFinder for that.

        Args:
            query_mass: Target mass to decompose
            charge: Charge state of the ion
             Default: 0
            ppm_error: Error tolerance in parts per million
             Default: 0.0
            mz_error: Error tolerance in daltons
             Default: 0.0
            min_counts: Minimum count for each element
             (or None for no minimum)
             Default: None
            max_counts: Maximum count for each element
             (or None for no limit)
             Default: None
            max_results: Maximum number of candidates to generate
             Default: 10000

        Returns:
            List of Formula objects representing possible elemental compositions
        """
        # Determine range of possible counts for each element
        max_counts, min_counts = self._populate_counts_based_on_user_input(
            max_counts,
            min_counts,
        )

        bounds, min_values = self._get_possible_element_count_bounds(
            max_counts,
            min_counts,
        )

        # Adjust mass depending on charge argument
        adjusted_mass = query_mass + (ELECTRON.mass * charge)

        # Calculate original mass range for final filtering
        original_min_mass, original_max_mass = self._populate_mass_range_based_on_user_input(
            mass=adjusted_mass,
            ppm_error=ppm_error,
            mz_error=mz_error,
        )

        # Calculate reduced mass range for decomposition search
        min_mass, max_mass = self._populate_mass_range_based_on_user_input(
            mass=adjusted_mass,
            ppm_error=ppm_error,
            mz_error=mz_error,
            min_counts=min_counts,
        )

        # Convert mass range to discretized units
        min_int, max_int = self._get_mass_range_as_integers(
            min_mass=max(0.0, min_mass),
            max_mass=max(0.0, max_mass),
        )

        # Decompose each integer mass in the discretized range
        results: list[Formula] = []

        for m in range(min_int, max_int + 1):
            decompositions = _decompose_integer_mass(
                ERT=self.ERT,
                elements=self.elements,
                mass=m,
                bounds=NumbaList(bounds),
                max_results=max_results,
            )

            # Convert decompositions to actual formulae, and apply checks
            for decomp in decompositions:

                # Apply minimum values
                for j in range(len(min_values)):
                    decomp[j] += min_values[j]

                if sum(decomp) == 0:
                    # Empty decomposition (idk how???)
                    continue

                # Convert decomp array into formula string
                formula_str = ''.join(
                    f"{symbol}{count}" for symbol, count in zip(
                        self.element_symbols, decomp
                    ) if count > 0
                )


                # Append charge
                formula_str = _append_charge(
                    formula_str,
                    charge,
                )

                # Create Formula object and calculate exact mass
                formula = Formula(formula_str)
                formula_mass = formula.monoisotopic_mass

                # Check if within desired range after conversion back to real
                if not original_min_mass <= formula_mass <= original_max_mass:
                    continue

                results.append(formula)

        # Return list of Formula objects (no filtering or error calculation)
        return results

    def _populate_mass_range_based_on_user_input(
        self,
        mass: float,
        ppm_error: Optional[float] = 0.0,
        mz_error: Optional[float] = 0.0,
        min_counts: Optional[dict[str, int]] = None,
    ) -> tuple[float, float]:
        """
        Given a mass and error range (ppm or Da),
        and the minimum/maximum element counts to consider,
        returns the lowest and highest masses to target.

        If both ppm_error and mz_error are given, will
        take the largest of the two

        Args:
            mass: Target mass to decompose
            ppm_error: Error in parts per million
            mz_error: Error in daltons

        Returns:
            min_mass: lowest viable mass
            max_mass: highest viable mass
        """
        if not ppm_error and not mz_error:
            raise ValueError(
                "Neither ppm_error nor mz_error arguments given"
            )

        error: float = max(
            mass * ppm_error / 1e6,
            mz_error,
        )

        min_mass: float = mass - error
        max_mass: float = mass + error

        # Adjust mass range based on minimum element counts
        if min_counts:
            for element in self.elements:
                if element.symbol in min_counts and min_counts[element.symbol] > 0:
                    reduce_by = element.mass * min_counts[element.symbol]
                    min_mass -= reduce_by
                    max_mass -= reduce_by

        return min_mass, max_mass

    def _get_possible_element_count_bounds(
        self,
        max_counts: Optional[ dict[str, int] ],
        min_counts: Optional[ dict[str, int] ],
    ) -> tuple[list[float], list[int]]:
        bounds: list[float] = [float('inf')] * len(self.elements)
        min_values = [0] * len(self.elements)

        # Set min and max values from dictionaries
        for i, element in enumerate(self.elements):
            if min_counts and element.symbol in min_counts:
                min_values[i] = min_counts[element.symbol]

            if max_counts and element.symbol in max_counts:
                bounds[i] = float(max_counts[element.symbol] - min_values[i])

        return bounds, min_values

    def _get_mass_range_as_integers(
        self,
        min_mass: float,
        max_mass: float,
    ) -> tuple[int, int]:
        """
        Convert mass range to integer range using pre-defined precision factor
        """
        from_int = math.ceil((1 + self.min_error) * min_mass / self.precision)
        to_int = math.floor((1 + self.max_error) * max_mass / self.precision)

        # Handle overflow
        if from_int > 9223372036854775807 or to_int > 9223372036854775807:
            raise ArithmeticError(
                f"Mass range ({min_mass} - {max_mass}) too large to "
                f"decompose with current precision: {self.precision}"
            )

        # Ensure valid range
        start = max(0, int(from_int))
        end = max(start, int(to_int))

        return start, end

    def _populate_counts_based_on_user_input(
        self,
        user_max_counts: Optional[dict[str, int]],
        user_min_counts: Optional[dict[str, int]],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Helper function that pre-populates min/max element count dicts
        depending on user input
        """
        # Set min counts
        final_min_counts: dict[str, int] = {
            e.symbol: 0 for e in self.elements
        }
        if user_min_counts:
            final_min_counts.update(**user_min_counts)

        # if user_min_counts is None:
        #     user_min_counts = {e.symbol: 0 for e in self.elements}
        # else:
        #     # Fill in any missing elements with 0
        #     for e in self.elements:
        #         if e.symbol not in user_min_counts:
        #             user_min_counts[e.symbol] = 0


        # Set max counts
        final_max_counts: dict[str, float] = {
            e.symbol: float('inf') for e in self.elements
        }
        if user_max_counts:
            final_max_counts.update(**user_max_counts)
        # if user_max_counts is None:
        #     user_max_counts = {e.symbol: float('inf') for e in self.elements}
        # else:
        #     # Fill in any missing elements with infinity
        #     for e in self.elements:
        #         if e.symbol not in user_max_counts:
        #             user_max_counts[e.symbol] = float('inf')


        return final_max_counts, final_min_counts


def get_element_most_abundant_mass(
    element: 'Element',
) -> float:
    """
    Given a molmass.Element instance, returns the
    mass of its most abundant isotope.

    This is the mass that is used for mass decomposition.

    Note: might make package unsuitable for very large molecules
    """
    isotopes: list['Isotope'] = sorted(
        element.isotopes.values(),
        key=lambda x: x.abundance,
        reverse=True,

    )
    return isotopes[0].mass

def _append_charge(
    formula_str: str,
    charge: int,
) -> str:
    """
    Given a formula string, returns a version with
    "+" or "-" appended X times, where X is `charge`

    i.e. 'C6H14O6', charge = 2, yields: 'C6H14O6++'
    """
    if charge == 0:
        return formula_str

    sign = "+" if charge > 0 else "-"

    output = formula_str
    for _ in range(abs(charge)):
        output = output + sign

    return output



