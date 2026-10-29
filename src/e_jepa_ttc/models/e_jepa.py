"""Public JEPA model names mapped to the audited implementations."""

from e_jepa_ttc.models.e_jepa_tubelet_lhr import EJEPATubeletLHR, EJEPATubeletLHRConfig
from e_jepa_ttc.models.object_jepa import ObjectCentricEventJEPA, ObjectJEPAConfig
from e_jepa_ttc.training.jepa import DenseTemporalJEPAPredictor

__all__ = [
    "DenseTemporalJEPAPredictor",
    "EJEPATubeletLHR",
    "EJEPATubeletLHRConfig",
    "ObjectCentricEventJEPA",
    "ObjectJEPAConfig",
]
