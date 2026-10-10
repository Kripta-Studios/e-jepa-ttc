"""Executable V13 reference operators; no dataset loader or trained weights included."""
from .phase import emitted_phase, phase_to_ttc, ttc_to_phase
from .model import EventModel, RGBExtension, HeadOnlyRGBBridge
__all__ = ['EventModel', 'RGBExtension', 'HeadOnlyRGBBridge', 'emitted_phase', 'phase_to_ttc', 'ttc_to_phase']
