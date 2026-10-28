"""Keep the selected replay profile separate from numerical control primitives."""

import inspect

from operational.train40_runtime.defaults import DEFAULT_PROFILE
from operational.train40_runtime.multiobject import MultiObjectTrain40Service
from operational.train40_runtime.producers import ProducerRuntime
from operational.train40_runtime.rgb_raw_ring import freeze_rgb_frame


def test_selected_profile_uses_exact_transport_without_enabling_experimental_pixels():
    profile = DEFAULT_PROFILE
    assert profile.backend == "prefix" and profile.event_normalization == "topk"
    assert profile.event_query_graphs and profile.parallel_branches
    assert not profile.reuse_event_encodings and profile.rgb_object_chunk_size == 1
    assert not profile.rgb_uint8 and not profile.shared_rgb_images
    assert profile.overlap_event_with_rgb_preparation
    assert profile.transport_reuse == "reverse-summaries"
    assert profile.paced_raw_source == "memory"
    assert profile.paced_prime_native_histories and profile.paced_rgb_feature_cache


def test_service_inherits_profile_but_control_primitive_and_sha_v1_remain_original():
    service = inspect.signature(MultiObjectTrain40Service).parameters
    assert service["overlap_event_with_rgb_preparation"].default is True
    assert service["transport_reuse"].default == DEFAULT_PROFILE.transport_reuse
    assert service["rgb_crop_cache"].default is False
    assert inspect.signature(ProducerRuntime).parameters["transport_reuse"].default == "original"
    assert inspect.signature(freeze_rgb_frame).parameters["hash_backend"].default is None
