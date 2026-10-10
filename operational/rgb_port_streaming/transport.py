"""Instance-local vectorized transport with independent Dynamo code identities."""

import types
from typing import cast

from torch import nn

from operational.streaming_revision.kernels import correlation


def install_transport(model: nn.Module) -> None:
    """Clone the code object as well as globals to isolate A5/C2F compiled guards."""
    method = cast(types.MethodType, model._forward_impl)
    original = cast(types.FunctionType, method.__func__)
    namespace = dict(original.__globals__, local_correlation_match=correlation)
    replacement = types.FunctionType(
        original.__code__.replace(),
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    replacement.__kwdefaults__ = original.__kwdefaults__
    model.__dict__["_forward_impl"] = types.MethodType(replacement, model)
