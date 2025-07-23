"""
Physics module for the URP Map project.

This module provides functionality for handling the plasma physics calculations.
"""

from .kappa import (
    KappaParams, 
    kappa_distribution, directional_flux, omnidirectional_flux
)

__all__ = [
    "KappaParams",
    "kappa_distribution",
    "directional_flux",
    "omnidirectional_flux"
]