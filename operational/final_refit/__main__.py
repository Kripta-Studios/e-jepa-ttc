"""CLI for registering, inspecting and running the isolated final refit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .recipe import build_recipe, load_recipe, stage, write_recipe
from .training import fit_cached_stage


def _split_pairs(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate binding: {name!r}")
        result[name] = value
    return result


def _path_pairs(values: Sequence[str]) -> dict[str, Path]:
    return {name: Path(value).resolve(strict=True) for name, value in _split_pairs(values).items()}


def _register(args: argparse.Namespace) -> int:
    source_values = _path_pairs(args.source)
    parent_values = _split_pairs(args.external_parent)
    recipe = build_recipe(
        source_files=source_values,
        external_parent_sha256=parent_values,
        native_train40_rows_sha256=args.native_train40_rows_sha256,
        ordered_train40_identity_sha256=args.ordered_train40_identity_sha256,
    )
    write_recipe(args.output.resolve(), recipe)
    print(json.dumps({"status": "REGISTERED", "recipe_sha256": recipe.recipe_sha256}))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe.resolve(strict=True))
    print(
        json.dumps(
            {
                "status": "VALID",
                "recipe_sha256": recipe.recipe_sha256,
                "queries": recipe.train_queries,
                "groups": recipe.train_groups,
                "new_scientific_updates": sum(item.updates for item in recipe.stages),
                "stages": [
                    {"id": item.stage_id, "kind": item.kind, "updates": item.updates}
                    for item in recipe.stages
                ],
            },
            indent=2,
        )
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    recipe_path = args.recipe.resolve(strict=True)
    recipe = load_recipe(recipe_path)
    selected = stage(recipe, args.stage)
    if selected.kind == "rgb_producer":
        raise ValueError("RGB producers are run through operational.final_refit.producers")
    bindings = _path_pairs(args.parent_receipt)
    receipt = fit_cached_stage(
        recipe=recipe,
        recipe_path=recipe_path,
        stage=selected,
        cache_path=args.cache.resolve(strict=True),
        parent_receipts=bindings,
        run_root=args.run.resolve(),
        device_name=args.device,
        max_updates_this_call=args.max_updates,
        resident_cache=not args.stream_cache,
        technical=args.technical,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "COMPLETE" else 3


def parser() -> argparse.ArgumentParser:
    """Construct the command parser without touching data or CUDA."""
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="Freeze the final recipe before training")
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    register.add_argument("--external-parent", action="append", default=[], metavar="NAME=SHA256")
    register.add_argument("--native-train40-rows-sha256", required=True)
    register.add_argument("--ordered-train40-identity-sha256", required=True)
    register.set_defaults(function=_register)
    inspect = commands.add_parser("inspect", help="Validate and print a zero-update recipe")
    inspect.add_argument("--recipe", type=Path, required=True)
    inspect.set_defaults(function=_inspect)
    train = commands.add_parser("train", help="Run or resume one cached final-refit stage")
    train.add_argument("--recipe", type=Path, required=True)
    train.add_argument("--stage", required=True)
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--parent-receipt", action="append", default=[], metavar="NAME=PATH")
    train.add_argument("--run", type=Path, required=True)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--max-updates", type=int)
    train.add_argument("--stream-cache", action="store_true")
    train.add_argument(
        "--technical",
        action="store_true",
        help="Write under technical/; requires --max-updates and never forms an endpoint",
    )
    train.set_defaults(function=_train)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    """Run one final-refit CLI command."""
    args = parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
