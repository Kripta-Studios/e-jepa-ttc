.PHONY: setup lint test check smoke-data scan-data validate-data index-data split-data cache-voxel train-base pretrain-jepa architecture-validate architecture-smoke architecture-screen

setup:
	uv sync --all-groups --no-editable

lint:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .

test:
	uv run --no-sync pytest

check: lint test

smoke-data:
	uv run --no-sync e-jepa-ttc synthetic generate --output tests/fixtures/synthetic_events.h5 --windows 64 --seed 7

scan-data:
	uv run --no-sync e-jepa-ttc data scan --root datasets/evttc --output data/manifests/evttc_local.yaml

validate-data:
	uv run --no-sync e-jepa-ttc data validate --manifest data/manifests/evttc_local.yaml

index-data:
	uv run --no-sync e-jepa-ttc data index --manifest data/manifests/evttc_local.yaml --output data/cache/evttc_index.json

split-data:
	uv run --no-sync e-jepa-ttc split create --manifest data/manifests/evttc_local.yaml --output data/splits/evttc_local.yaml --seed 42

cache-voxel:
	uv run --no-sync e-jepa-ttc cache voxel --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --index data/cache/evttc_index.json --output artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --width 160 --height 90 --bins 5 --no-normalize --metadata-channels

train-base:
	uv run --no-sync e-jepa-ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/base_seed7 --epochs 60 --batch-size 32 --learning-rate 0.0003 --seed 7 --device auto

pretrain-jepa:
	uv run --no-sync e-jepa-ttc pretrain jepa --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/jepa_seed7 --epochs 80 --batch-size 32 --learning-rate 0.0005 --seed 7 --device auto --pretrain-splits train --validation-splits validation --temporal-horizons-ms 20 60 100 240 500

architecture-validate:
	powershell -ExecutionPolicy Bypass -File scripts/run_evttc_architecture_selection.ps1 -Mode Validate

architecture-smoke:
	powershell -ExecutionPolicy Bypass -File scripts/run_evttc_architecture_selection.ps1 -Mode Smoke -Resume

architecture-screen:
	powershell -ExecutionPolicy Bypass -File scripts/run_evttc_architecture_selection.ps1 -Mode Screen -Resume
