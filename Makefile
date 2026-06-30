.PHONY: setup test lint smoke-data scan-data validate-data index-data split-data train-baseline clean

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

train-baseline:
	uv run --no-sync e-jepa-ttc baseline trivial --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --output artifacts/metrics/trivial_baseline.json

clean:
	python -m compileall -q src tests
