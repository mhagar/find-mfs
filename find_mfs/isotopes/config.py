"""
This module provides config dataclasses for different isotope
pattern matching strategies
"""
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np


@dataclass
class SingleEnvelopeMatch:
    """
    Configuration for matching against a single isotope envelope measurement.

    This approach compares the predicted isotope pattern against a single
    observed envelope, checking that each peak falls within specified
    m/z and intensity tolerances.

    This class can be passed to FormulaFinder or FormulaSearchResults to
    initiate isotope pattern matching

    Attributes:
        envelope: 2D numpy array of [m/z, intensity] pairs representing
            the observed isotope envelope. Intensities will be normalized
            such that the base peak is 1.0

        mz_tolerance_da: Maximum m/z difference (in Da) for a predicted peak
            to match an observed peak. Either this or mz_tolerance_ppm must
            be specified.
            Default: None

        mz_tolerance_ppm: Maximum m/z difference in ppm for a predicted peak
            to match an observed peak. Either this or mz_tolerance must be
            specified.
            Default: None

        intensity_tolerance: Maximum intensity difference for a predicted
            peak to match an observed peak.
            Default: 0.1

        simulated_mz_tolerance: Resolution for simulating theoretical
            envelopes. Peaks closer than this will be combined.
            Default: 0.05

        simulated_intensity_threshold: Minimum relative intensity threshold
            for including peaks in the simulated envelope.
            Default: 0.001

    Example:
        >>> observed = np.array(
        ...     [[180.063, 1.0],
        ...      [181.067, 0.11]]
        ... )
        >>> config = SingleEnvelopeMatch(
        ...     envelope=observed,
        ...     # mz_tolerance_da=0.01,  # Tolerance in Da or ppm
        ...     mz_tolerance_ppm=4.0,
        ...     intensity_tolerance=0.05
        ... )
    """
    envelope: np.ndarray
    mz_tolerance_da: Optional[float] = None
    mz_tolerance_ppm: Optional[float] = None
    intensity_tolerance: float = 0.1
    simulated_mz_tolerance: float = 0.05
    simulated_intensity_threshold: float = 0.001

    def __post_init__(self):
        """
        Validate configuration parameters on initializing
        """
        if self.envelope.ndim != 2 or self.envelope.shape[1] != 2:
            raise ValueError(
                f"envelope must be a 2D array with shape (n, 2), "
                f"got shape {self.envelope.shape}"
            )

        # Validate exactly one tolerance type is specified
        if self.mz_tolerance_da is None and self.mz_tolerance_ppm is None:
            raise ValueError(
                "Either mz_tolerance or mz_tolerance_ppm must be specified"
            )

        if self.mz_tolerance_da is not None and self.mz_tolerance_ppm is not None:
            raise ValueError(
                "Specify either mz_tolerance or mz_tolerance_ppm, not both"
            )

        # Validate tolerance values are positive
        if self.mz_tolerance_da is not None and self.mz_tolerance_da <= 0:
            raise ValueError(
                f"mz_tolerance must be positive, got {self.mz_tolerance_da}"
            )

        if self.mz_tolerance_ppm is not None and self.mz_tolerance_ppm <= 0:
            raise ValueError(
                f"mz_tolerance_ppm must be positive, got {self.mz_tolerance_ppm}"
            )

        if self.intensity_tolerance <= 0:
            raise ValueError(
                f"intensity_tolerance must be positive, "
                f"got {self.intensity_tolerance}"
            )

        # Normalize envelope
        self.envelope[:, 1] = self.envelope[:, 1]/self.envelope[:, 1].max()


@dataclass
class MultiEnvelopeMatch:
    """
    ***PLACE HOLDER***
    Will implement an alternative isotope pattern strategy in the future
    """


# Type alias for any isotope matching config
IsotopeMatchConfig = Union[SingleEnvelopeMatch, MultiEnvelopeMatch]
