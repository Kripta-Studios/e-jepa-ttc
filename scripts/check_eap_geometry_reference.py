"""Re-execute the pinned official bundle formula and render a synthetic diagnostic.

No RGB, event or label data are read. This checks the original source convention;
it deliberately does not assert a conversion for the Hugging Face release.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from e_jepa_ttc.detection.geometry_contract import original_bundle_corners


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-visualizer", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/detection/eap_original_visualizer_corners.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    source = args.official_visualizer.read_bytes()
    if hashlib.sha256(source).hexdigest() != fixture["source_sha256"]:
        raise ValueError("Official visualizer bytes differ from the reviewed fixed source.")
    tree = ast.parse(source.decode())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bbox_to_corner_ego"
    )
    scope = {"np": np, "math": math}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(args.official_visualizer), "exec"),
        scope,
    )
    transform = np.array([[0, -1, 0, 1], [0, 0, -1, 2], [1, 0, 0, 3], [0, 0, 0, 1]])
    intrinsic = np.array([[100, 0, 20], [0, 100, 10], [0, 0, 1]])
    checks = []
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="640" viewBox="0 0 1000 640">',
        '<rect width="1000" height="640" fill="#f6f8fa"/>',
        '<g font-family="sans-serif" fill="#142b44">',
        '<text x="36" y="35" font-size="23">'
        "eAP: independent original-bundle geometry fixture</text>",
        '<text x="36" y="62" font-size="14">Synthetic cuboid and camera. '
        "This is not a real sensor reprojection.</text>",
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, case in enumerate(fixture["cases"]):
        box = case["bbox_bundle_xyzlhw_yaw"]
        official = scope["bbox_to_corner_ego"](*box)
        actual = original_bundle_corners(box)
        np.testing.assert_allclose(actual, official, atol=1e-6, rtol=0)
        np.testing.assert_allclose(official, case["corners_ego_m"], atol=1e-6, rtol=0)
        camera = (transform @ np.column_stack([official, np.ones(8)]).T).T[:, :3]
        projected = (intrinsic @ camera.T).T
        pixels = projected[:, :2] / projected[:, 2:3]
        checks.append(
            {
                "yaw_rad": box[6],
                "max_corner_error_m": float(np.max(np.abs(actual - official))),
                "camera_depth_min_m": float(camera[:, 2].min()),
                "projection_px": pixels.tolist(),
            }
        )
        x, y = 35 + (i % 2) * 490, 86 + (i // 2) * 250
        svg.append(
            f'<rect x="{x}" y="{y}" width="455" height="225" rx="8" fill="white" stroke="#cbd5e1"/>'
        )
        svg.append(f'<text x="{x + 16}" y="{y + 27}" font-size="16">yaw = {box[6]:.6f} rad</text>')
        # Fixed pixel axes across panels retain scale; no independent fitting of boxes.
        screen = np.column_stack(
            [x + 220 + (pixels[:, 0] - 20) * 3, y + 120 + (pixels[:, 1] - 10) * 3]
        )
        for a, b in edges:
            svg.append(
                f'<line x1="{screen[a, 0]:.3f}" y1="{screen[a, 1]:.3f}" '
                f'x2="{screen[b, 0]:.3f}" y2="{screen[b, 1]:.3f}" '
                'stroke="#0369a1" stroke-width="2"/>'
            )
        svg.append(
            f'<text x="{x + 16}" y="{y + 207}" font-size="12">'
            "l=6 m; h=2 m; w=4 m; center=(10,3,2) m</text>"
        )
    svg.extend(
        [
            '<text x="36" y="607" font-size="14">'
            "PASS: source corners and explicit ego-to-camera transform. "
            "BLOCKED: HF bridge and real RGB-event alignment.</text>",
            "</g></svg>",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "g0_original_geometry.svg").write_text("\n".join(svg), encoding="utf-8")
    result = {
        "status": "PASS_ORIGINAL_FORMAT_ONLY",
        "hf_geometry_verified": False,
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "official_source_sha256": fixture["source_sha256"],
        "source_commit": fixture["source_commit"],
        "transform_ego_to_camera": transform.tolist(),
        "intrinsic": intrinsic.tolist(),
        "cases": checks,
    }
    (args.output_dir / "g0_original_geometry.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": result["status"], "cases": len(checks), "hf_geometry_verified": False}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
