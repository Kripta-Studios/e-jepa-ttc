"""Versioned public name for the factorized Tubelet LHR implementation."""

from e_jepa_ttc.models.highres_factorized import (
    EJEPATubeletLHR,
    EJEPATubeletLHRConfig,
    EJEPATubeletLHROutput,
    HighResFeatures,
    PatchGeometry,
    TheoreticalOOMError,
)

EJEPATubeletLHRv4 = EJEPATubeletLHR

__all__ = [
    "EJEPATubeletLHR",
    "EJEPATubeletLHRConfig",
    "EJEPATubeletLHRv4",
    "EJEPATubeletLHROutput",
    "HighResFeatures",
    "PatchGeometry",
    "TheoreticalOOMError",
]
