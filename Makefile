.PHONY: setup test lint smoke-data scan-data validate-data index-data split-data train-baseline baseline-trivial baseline-geometric baseline-event-rate cache-voxel train-tiny-cnn clean

setup:
	uv sync --all-groups --no-editable

test:
	uv run --no-sync pytest

lint:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .

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

train-baseline: baseline-trivial

baseline-trivial:
	uv run --no-sync e-jepa-ttc baseline trivial --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --output artifacts/metrics/trivial_baseline.json

baseline-geometric:
	uv run --no-sync e-jepa-ttc baseline geometric --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --output artifacts/metrics/geometric_baseline.json

baseline-event-rate:
	uv run --no-sync e-jepa-ttc baseline event-rate --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --index data/cache/evttc_index.json --output artifacts/metrics/event_rate_baseline.json

cache-voxel:
	uv run --no-sync e-jepa-ttc cache voxel --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --index data/cache/evttc_index.json --output artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --width 160 --height 90 --bins 5 --no-normalize --metadata-channels

train-tiny-cnn:
	uv run --no-sync e-jepa-ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_seed7 --epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto

clean:
	python -m compileall -q src tests

