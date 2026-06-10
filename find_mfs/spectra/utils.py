"""Utilities for loading spectra from common file formats."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .envelopes import SpectrumArray, to_spec_arr


@dataclass(slots=True)
class MGFSpectrum:
    """
    One spectrum block from an MGF file
    """
    spec_arr: SpectrumArray
    metadata: dict[str, str] = field(default_factory=dict)


def read_mgf(
    path: str | Path
) -> list[MGFSpectrum]:
    """
    Parse an MGF file and return one MGFSpectrum per BEGIN IONS block.

    Metadata keys (PEPMASS, CHARGE, etc.) are stored as strings in .metadata.
    Peak lines are "mz intensity" pairs.
    """
    spectra: list[MGFSpectrum] = []
    path = Path(path)

    with path.open() as fh:
        in_block = False
        metadata: dict[str, str] = {}
        mzs: list[float] = []
        intensities: list[float] = []

        for line in fh:
            line = line.strip()
            if not line:
                continue

            if line == "BEGIN IONS":
                in_block = True
                metadata = {}
                mzs = []
                intensities = []
            elif line == "END IONS":
                if mzs:
                    spectra.append(MGFSpectrum(
                        spec_arr=to_spec_arr(mzs, intensities),
                        metadata=metadata,
                    ))
                in_block = False
            elif in_block:
                if "=" in line:
                    key, _, value = line.partition("=")
                    metadata[key] = value
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        mzs.append(float(parts[0]))
                        intensities.append(float(parts[1]))

    return spectra
