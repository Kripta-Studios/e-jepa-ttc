"""Optional two-component velocity head for the portable MonoDGP variant.

The extension predicts the first two released eAP velocity components. Their
physical frame semantics remain a separate data-contract question: this module
does not interpret them as ego velocity, radial velocity, or TTC.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional


def matched_velocity_l1(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Return summed component L1 per valid matched object.

    ``prediction`` and ``target`` have shape ``[M, 2]`` and ``valid`` has
    shape ``[M]``. Invalid target rows are never read. The reduction sums the
    absolute X/Y component errors and divides by the number of valid objects,
    so its unit is metres per second per matched object. With no valid objects,
    it returns a differentiable zero connected to ``prediction``.
    """
    if prediction.ndim != 2 or prediction.shape[1:] != (2,):
        raise ValueError("Velocity prediction must have shape [M,2].")
    if target.shape != prediction.shape:
        raise ValueError("Velocity target must match prediction shape [M,2].")
    if valid.ndim != 1 or valid.shape[0] != prediction.shape[0]:
        raise ValueError("Velocity validity mask must have shape [M].")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("Velocity prediction and target must be floating point.")
    if valid.dtype is not torch.bool:
        raise TypeError("Velocity validity mask must be boolean.")
    if target.device != prediction.device or valid.device != prediction.device:
        raise ValueError("Velocity prediction, target, and mask must share one device.")
    if not torch.isfinite(prediction).all().item():
        raise ValueError("Velocity predictions must be finite.")
    if not valid.any().item():
        return prediction.sum() * 0.0
    valid_target = target[valid]
    if not torch.isfinite(valid_target).all().item():
        raise ValueError("Valid velocity targets must be finite.")
    count = valid.sum().to(dtype=prediction.dtype)
    return functional.l1_loss(prediction[valid], valid_target, reduction="sum") / count


def apply_velocity_extension(edit: Callable[[str, str, str], None]) -> None:
    """Describe deterministic edits that add the opt-in MonoDGP velocity head.

    ``edit`` receives ``(relative_path, unique_old_anchor, replacement)``. The
    caller owns source materialization, uniqueness checks, patch generation,
    and full-tree hashing. Only ``monodgp.py`` is changed by this extension.
    """
    path = "lib/models/monodgp/monodgp.py"

    edit(
        path,
        "from .position_encoding import PositionEmbeddingCamRay\n",
        "from .position_encoding import PositionEmbeddingCamRay\n"
        "from e_jepa_ttc.detection.velocity_extension import matched_velocity_l1\n",
    )
    edit(
        path,
        "                  aux_loss=True, with_box_refine=False, init_box=False, group_num=11):",
        "                  aux_loss=True, with_box_refine=False, init_box=False, group_num=11,\n"
        "                  predict_velocity=False):",
    )
    edit(
        path,
        "        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation\n",
        "        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation\n"
        "        self.predict_velocity = bool(predict_velocity)\n"
        "        if self.predict_velocity:\n"
        "            self.velocity_embed = MLP(hidden_dim, hidden_dim, 2, 2)\n",
    )
    edit(
        path,
        "        num_pred = det3d_transformer.decoder.num_layers + 1\n",
        "        num_pred = det3d_transformer.decoder.num_layers + 1\n"
        "        velocity_num_pred = det3d_transformer.decoder.num_layers\n",
    )
    edit(
        path,
        "            self.depthaware_transformer.decoder.bbox_embed = None\n\n\n    def forward",
        "            self.depthaware_transformer.decoder.bbox_embed = None\n\n"
        "        if self.predict_velocity:\n"
        "            self.velocity_embed = _get_clones(\n"
        "                self.velocity_embed, velocity_num_pred\n"
        "            )\n\n\n"
        "    def forward",
    )
    edit(
        path,
        "        outputs_angles = []\n\n        for lvl in range(hs.shape[0]):",
        "        outputs_angles = []\n"
        "        outputs_velocities = [] if self.predict_velocity else None\n\n"
        "        for lvl in range(hs.shape[0]):",
    )
    edit(
        path,
        "            outputs_angles.append(outputs_angle)\n\n"
        "        outputs_coord = torch.stack(outputs_coords)",
        "            outputs_angles.append(outputs_angle)\n\n"
        "            if outputs_velocities is not None:\n"
        "                outputs_velocities.append(self.velocity_embed[lvl](hs[lvl]))\n\n"
        "        outputs_coord = torch.stack(outputs_coords)",
    )
    edit(
        path,
        "        outputs_angle = torch.stack(outputs_angles)\n  \n        out = dict()",
        "        outputs_angle = torch.stack(outputs_angles)\n"
        "        outputs_velocity = (\n"
        "            torch.stack(outputs_velocities) if outputs_velocities is not None else None\n"
        "        )\n  \n"
        "        out = dict()",
    )
    edit(
        path,
        "        out['pred_angle'] = outputs_angle[-1]\n        out['pred_depth_map_logits']",
        "        out['pred_angle'] = outputs_angle[-1]\n"
        "        if outputs_velocity is not None:\n"
        "            out['pred_velocity'] = outputs_velocity[-1]\n"
        "        out['pred_depth_map_logits']",
    )
    edit(
        path,
        "                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, "
        "outputs_depth) ",
        "                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle,\n"
        "                outputs_depth,\n"
        "                outputs_velocity) ",
    )
    edit(
        path,
        "    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, "
        "outputs_angle, outputs_depth):\n"
        "        # this is a workaround to make torchscript happy, as torchscript\n"
        "        # doesn't support dictionary with non-homogeneous values, such\n"
        "        # as a dict having both a Tensor and a list.\n"
        "        return [{'pred_logits': a, 'pred_boxes': b, \n"
        "                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}\n"
        "                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],\n"
        "                                         outputs_3d_dim[:-1], outputs_angle[:-1], "
        "outputs_depth[:-1])]",
        "    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle,\n"
        "                      outputs_depth, outputs_velocity):\n"
        "        # this is a workaround to make torchscript happy, as torchscript\n"
        "        # doesn't support dictionary with non-homogeneous values.\n"
        "        auxiliary = [\n"
        "            {'pred_logits': a, 'pred_boxes': b, 'pred_3d_dim': c,\n"
        "             'pred_angle': d, 'pred_depth': e}\n"
        "            for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],\n"
        "                                     outputs_3d_dim[:-1], outputs_angle[:-1],\n"
        "                                     outputs_depth[:-1])\n"
        "        ]\n"
        "        if outputs_velocity is not None:\n"
        "            for item, velocity in zip(auxiliary, outputs_velocity[:-1]):\n"
        "                item['pred_velocity'] = velocity\n"
        "        return auxiliary",
    )
    edit(
        path,
        "    def loss_dims(self, outputs, targets, indices, num_boxes):  \n",
        "    def loss_velocities(self, outputs, targets, indices, num_boxes):\n"
        "        del num_boxes\n"
        "        idx = self._get_src_permutation_idx(indices)\n"
        "        prediction = outputs['pred_velocity'][idx]\n"
        "        target = torch.cat([\n"
        "            item['velocity'][target_index]\n"
        "            for item, (_, target_index) in zip(targets, indices)\n"
        "        ], dim=0)\n"
        "        valid = torch.cat([\n"
        "            item['velocity_valid'][target_index]\n"
        "            for item, (_, target_index) in zip(targets, indices)\n"
        "        ], dim=0)\n"
        "        return {'loss_velocity': matched_velocity_l1(prediction, target, valid)}\n\n"
        "    def loss_dims(self, outputs, targets, indices, num_boxes):  \n",
    )
    edit(
        path,
        "            'depths': self.loss_depths,\n            'dims': self.loss_dims,",
        "            'depths': self.loss_depths,\n"
        "            'velocities': self.loss_velocities,\n"
        "            'dims': self.loss_dims,",
    )
    edit(
        path,
        "        init_box=cfg['init_box'],\n        group_num=cfg['group_num'])",
        "        init_box=cfg['init_box'],\n"
        "        group_num=cfg['group_num'],\n"
        "        predict_velocity=cfg.get('predict_velocity', False))",
    )
    edit(
        path,
        "    weight_dict['loss_region'] = cfg['region_loss_coef']\n    \n    if cfg['aux_loss']:",
        "    weight_dict['loss_region'] = cfg['region_loss_coef']\n"
        "    if cfg.get('predict_velocity', False):\n"
        "        weight_dict['loss_velocity'] = cfg.get('velocity_loss_coef', 1.0)\n"
        "    \n"
        "    if cfg['aux_loss']:",
    )
    edit(
        path,
        "    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', "
        "'angles', 'center', 'depth_map', 'region']\n\n"
        "    criterion = SetCriterion(",
        "    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', "
        "'angles', 'center', 'depth_map', 'region']\n"
        "    if cfg.get('predict_velocity', False):\n"
        "        losses.append('velocities')\n\n"
        "    criterion = SetCriterion(",
    )
