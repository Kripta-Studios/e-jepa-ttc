"""Consolidated finite-tensor validation with an exact historical failure path."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator

import torch
from torch import Tensor

HostDecision = Callable[[Tensor], bool]


def require_finite_reference(value: object, name: str = "output") -> None:
    """Historical recursive validator retained as the exact failure oracle."""
    if isinstance(value, Tensor) and not torch.isfinite(value).all().item():
        raise ValueError(f"Nonfinite tensor: {name}")
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_reference(item, f"{name}.{key}")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_reference(item, f"{name}[{index}]")


def _tensor_leaves(value: object, name: str) -> Iterator[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield name, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_leaves(item, f"{name}.{key}")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensor_leaves(item, f"{name}[{index}]")


def _default_host_decision(flag: Tensor) -> bool:
    return bool(flag.item())


def require_finite_fused(
    value: object,
    name: str = "output",
    *,
    host_decision: HostDecision = _default_host_decision,
) -> None:
    """Reject nonfinite tensor leaves using one host decision per device.

    Dense strided tensor leaves on CPU/CUDA are reduced on their own device and
    device flags are combined without moving tensor payloads. CPU tensors form
    their own group; each CUDA device forms a separate group. Empty structures
    make no host decision. Other layouts/devices delegate entirely to the
    historical validator before fused reductions begin.

    If any group fails, ``require_finite_reference`` reruns the historical
    traversal so the first offender, name formatting and ``ValueError`` remain
    identical. The fast success path never removes validation.
    """
    leaves = list(_tensor_leaves(value, name))
    if any(
        tensor.layout is not torch.strided or tensor.device.type not in {"cpu", "cuda"}
        for _, tensor in leaves
    ):
        require_finite_reference(value, name)
        return

    groups: OrderedDict[torch.device, list[Tensor]] = OrderedDict()
    for _, tensor in leaves:
        groups.setdefault(tensor.device, []).append(tensor)

    for tensors in groups.values():
        try:
            flags = [torch.isfinite(tensor).all() for tensor in tensors]
            device_flag = flags[0] if len(flags) == 1 else torch.stack(flags).all()
            passed = host_decision(device_flag)
        except Exception:  # noqa: BLE001 - reference must own unsupported-tensor behavior.
            require_finite_reference(value, name)
            raise
        if not passed:
            require_finite_reference(value, name)
            raise RuntimeError("Fused finite decision failed but reference traversal passed")
