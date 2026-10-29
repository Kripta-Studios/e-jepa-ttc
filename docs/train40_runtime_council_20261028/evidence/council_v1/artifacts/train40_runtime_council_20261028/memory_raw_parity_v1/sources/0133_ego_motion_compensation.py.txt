"""Explicit causal rotational and translational camera-motion compensation."""

from __future__ import annotations

import torch
from torch import nn


class CameraYawDerotator(nn.Module):
    """De-rotate past box rays using measured yaw rate in radians/second.

    No neural TTC correction is produced: navigation can alter the physical
    geometric inputs but cannot directly become a scenario-to-TTC shortcut.
    Intrinsics are normalized as ``fx/W, fy/H, cx/W, cy/H``.

    This is deliberately not full ego-motion compensation. EvTTC GNSS
    velocity is documented in the world frame; translating it into camera
    optical flow also requires an audited world-to-camera transform and depth.
    """

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        if action_dim < 8:
            raise ValueError("EvTTC ego actions must include yaw rate at index seven.")
        self.action_dim = action_dim

    def forward(
        self,
        boxes_xyxy: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor,
        times_s: torch.Tensor,
        *,
        intrinsics_normalized: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return yaw-aligned boxes and accumulated angles to the current time."""

        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError("actions must have shape [B,T,action_dim].")
        if valid_mask.shape != actions.shape[:2]:
            raise ValueError("valid_mask must have shape [B,T].")
        if boxes_xyxy.ndim != 4 or boxes_xyxy.shape[:2] != actions.shape[:2]:
            raise ValueError("boxes_xyxy must have shape [B,T,O,4].")
        batch, steps = actions.shape[:2]
        times = _times(times_s, batch=batch, steps=steps)
        angle = _accumulated_yaw(actions, valid_mask, times)
        intrinsics = _intrinsics(
            boxes_xyxy,
            intrinsics_normalized,
            batch=batch,
        )
        corners = _corners(boxes_xyxy)
        fx, fy, cx, cy = (value[:, None, None, None] for value in intrinsics.unbind(dim=-1))
        ray_x = (corners[..., 0] - cx) / fx.clamp_min(1e-6)
        ray_y = (corners[..., 1] - cy) / fy.clamp_min(1e-6)
        cosine = angle.cos()[:, :, None, None]
        sine = angle.sin()[:, :, None, None]
        rotated_x = cosine * ray_x + sine
        rotated_z = -sine * ray_x + cosine
        aligned_x = rotated_x / rotated_z.clamp_min(1e-4)
        aligned_y = ray_y / rotated_z.clamp_min(1e-4)
        pixel_x = aligned_x * fx + cx
        pixel_y = aligned_y * fy + cy
        aligned = torch.stack(
            (
                pixel_x.amin(dim=-1),
                pixel_y.amin(dim=-1),
                pixel_x.amax(dim=-1),
                pixel_y.amax(dim=-1),
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        valid_boxes = valid_mask[:, :, None, None]
        return torch.where(valid_boxes, aligned, boxes_xyxy), angle


class CameraEgoMotionCompensator(nn.Module):
    """Warp past boxes into the current camera pose and expose ego closing rate.

    Actions use the audited EvTTC contract
    ``[speed, vx, vy, vz, ax, ay, az, yaw_rate]`` in the event-camera optical
    frame. The warp is causal and uses only context measurements. Depth is
    required because translational image motion is not observable from camera
    velocity alone; callers must therefore declare whether depth is measured
    (oracle/teacher) or predicted.
    """

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        if action_dim < 8:
            raise ValueError("EvTTC ego actions must include camera velocity and yaw rate.")
        self.action_dim = action_dim

    def forward(
        self,
        boxes_xyxy: torch.Tensor,
        depth_history_m: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor,
        times_s: torch.Tensor,
        *,
        intrinsics_normalized: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ego-warped boxes, yaw, translation and current ego inverse TTC."""

        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError("actions must have shape [B,T,action_dim].")
        if valid_mask.shape != actions.shape[:2]:
            raise ValueError("valid_mask must have shape [B,T].")
        if boxes_xyxy.ndim != 4 or boxes_xyxy.shape[:2] != actions.shape[:2]:
            raise ValueError("boxes_xyxy must have shape [B,T,O,4].")
        if depth_history_m.shape != boxes_xyxy.shape[:3]:
            raise ValueError("depth_history_m must have shape [B,T,O].")
        batch, steps = actions.shape[:2]
        times = _times(times_s, batch=batch, steps=steps)
        intrinsics = _intrinsics(
            boxes_xyxy,
            intrinsics_normalized,
            batch=batch,
        )
        angle = _accumulated_yaw(actions, valid_mask, times)
        intervals = (times[:, 1:] - times[:, :-1]).clamp_min(0.0)
        interval_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        increments = (
            actions[:, 1:, 1:4] * intervals[..., None] * interval_valid[..., None].to(actions.dtype)
        )
        increments_current = _rotate_optical_yaw(increments, angle[:, 1:])
        reverse_translation = torch.flip(
            torch.cumsum(torch.flip(increments_current, dims=(1,)), dim=1),
            dims=(1,),
        )
        translation = torch.cat(
            (reverse_translation, torch.zeros_like(reverse_translation[:, :1])),
            dim=1,
        )

        corners = _corners(boxes_xyxy)
        fx, fy, cx, cy = (value[:, None, None, None] for value in intrinsics.unbind(dim=-1))
        ray_x = (corners[..., 0] - cx) / fx.clamp_min(1e-6)
        ray_y = (corners[..., 1] - cy) / fy.clamp_min(1e-6)
        depth = depth_history_m[..., None].to(boxes_xyxy)
        points = torch.stack(
            (
                ray_x * depth,
                ray_y * depth,
                torch.ones_like(ray_x) * depth,
            ),
            dim=-1,
        )
        points = _rotate_optical_yaw(points, angle[:, :, None, None])
        points = points - translation[:, :, None, None]
        positive_depth = points[..., 2] > 1e-4
        projected_x = points[..., 0] / points[..., 2].clamp_min(1e-4) * fx + cx
        projected_y = points[..., 1] / points[..., 2].clamp_min(1e-4) * fy + cy
        aligned = torch.stack(
            (
                projected_x.amin(dim=-1),
                projected_y.amin(dim=-1),
                projected_x.amax(dim=-1),
                projected_y.amax(dim=-1),
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        depth_valid = torch.isfinite(depth_history_m) & (depth_history_m > 1e-4)
        valid_boxes = (valid_mask[:, :, None] & depth_valid & positive_depth.all(dim=-1))[..., None]
        aligned = torch.where(valid_boxes, aligned, boxes_xyxy)
        current_depth = depth_history_m[:, -1].to(actions).clamp_min(1e-4)
        ego_inverse_ttc = actions[:, -1, 3:4] / current_depth
        ego_inverse_ttc = torch.where(
            valid_mask[:, -1, None] & depth_valid[:, -1],
            ego_inverse_ttc,
            torch.zeros_like(ego_inverse_ttc),
        )
        return aligned, angle, translation, ego_inverse_ttc


def _times(times_s: torch.Tensor, *, batch: int, steps: int) -> torch.Tensor:
    if times_s.ndim == 1 and times_s.shape[0] == steps:
        return times_s[None].expand(batch, -1)
    if times_s.shape == (batch, steps):
        return times_s
    raise ValueError("times_s must have shape [T] or [B,T].")


def _intrinsics(
    reference: torch.Tensor,
    intrinsics_normalized: torch.Tensor | None,
    *,
    batch: int,
) -> torch.Tensor:
    if intrinsics_normalized is None:
        return reference.new_tensor((1.0, 1.0, 0.5, 0.5))[None].expand(batch, -1)
    if intrinsics_normalized.shape != (batch, 4):
        raise ValueError("intrinsics_normalized must have shape [B,4].")
    return intrinsics_normalized.to(reference)


def _accumulated_yaw(
    actions: torch.Tensor,
    valid_mask: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    intervals = (times[:, 1:] - times[:, :-1]).clamp_min(0.0)
    interval_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
    yaw_increment = actions[:, 1:, 7] * intervals * interval_valid.to(actions.dtype)
    reverse = torch.flip(
        torch.cumsum(torch.flip(yaw_increment, dims=(1,)), dim=1),
        dims=(1,),
    )
    return torch.cat((reverse, torch.zeros_like(reverse[:, :1])), dim=1)


def _rotate_optical_yaw(vectors: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    while angle.ndim < vectors.ndim - 1:
        angle = angle.unsqueeze(-1)
    cosine = angle.cos()
    sine = angle.sin()
    x, y, z = vectors.unbind(dim=-1)
    return torch.stack(
        (
            cosine * x + sine * z,
            y,
            -sine * x + cosine * z,
        ),
        dim=-1,
    )


def _corners(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes_xyxy.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((x0, y0), dim=-1),
            torch.stack((x1, y0), dim=-1),
            torch.stack((x1, y1), dim=-1),
            torch.stack((x0, y1), dim=-1),
        ),
        dim=-2,
    )


# Backward import compatibility for development checkpoints.
EgoMotionCompensator = CameraYawDerotator

__all__ = [
    "CameraEgoMotionCompensator",
    "CameraYawDerotator",
    "EgoMotionCompensator",
]
