"""
This module contains functions for calculating mass error
"""
from typing import Optional
from molmass import Formula, CompositionItem, ELEMENTS


def calc_error_ppm(
    observed_mz: float,
    predicted_mz: float,
) -> float:
    return (predicted_mz - observed_mz)/predicted_mz * 1e6


def calc_error_da(
    observed_mz: float,
    predicted_mz: float,
) -> float:
    """
    Calculate error in daltons (absolute mass difference)
    """
    return abs(predicted_mz - observed_mz)
