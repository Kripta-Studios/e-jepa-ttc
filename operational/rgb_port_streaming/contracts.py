"""Adapt the actual V13 producer mask contract without changing its values."""

from dataclasses import replace

from e_jepa_ttc.rgb_port.features import ProducerObservation

_ORIGINAL_VALIDATE = ProducerObservation.validate


def validate_observation(value: ProducerObservation) -> None:
    """The native known mask describes the current TTC, with shape [B].

    Validate all other fields with the retained validator. Expansion is used
    only in this validation copy; actual mask values/shapes remain unchanged.
    Historical [B,T] producers retain their original validation.
    """
    if value.known.shape == value.token128.shape[:-1]:
        checking = replace(value, known=value.known[..., None].expand_as(value.support))
        _ORIGINAL_VALIDATE(checking)
    else:
        _ORIGINAL_VALIDATE(value)


class NativeObservation(ProducerObservation):
    """Accept the actual current-prediction mask while preserving all sensor clocks."""

    def validate(self) -> None:
        validate_observation(self)
