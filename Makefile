.PHONY: setup lint test check smoke-data scan-data validate-data index-data split-data cache-voxel train-base train-baseline pretrain-jepa finetune-ttc garl-screen garl-full-dry-run garl-full evaluate demo architecture-validate architecture-smoke architecture-screen eap-analysis eap-full report

EAP_ROOT ?=
GARLTTC_ROOT ?=
GARL_SPLIT ?= data/splits/eap_pilot12_v1.json
FINETUNE_OUTPUT ?= artifacts/runs/e_jepa_tubelet_lhr_event
METRICS_JSON ?=
SPLIT ?= validation

setup:
	uv sync --locked --all-groups --no-editable

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

train-baseline: train-base

pretrain-jepa:
	uv run --no-sync e-jepa-ttc pretrain jepa --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/jepa_seed7 --epochs 80 --batch-size 32 --learning-rate 0.0005 --seed 7 --device auto --pretrain-splits train --validation-splits validation --temporal-horizons-ms 20 60 100 240 500

finetune-ttc:
	@test -n "$(EAP_ROOT)" || (echo "EAP_ROOT is required" && exit 2)
	@test -n "$(GARLTTC_ROOT)" || (echo "GARLTTC_ROOT is required" && exit 2)
	uv run --no-sync python scripts/run_e_jepa_garl_final.py --profile screen --stages train --eap-root "$(EAP_ROOT)" --garlttc-root "$(GARLTTC_ROOT)" --split "$(GARL_SPLIT)" --output-root "$(FINETUNE_OUTPUT)" --device auto

garl-screen: finetune-ttc

garl-full-dry-run:
	@test -n "$(EAP_ROOT)" || (echo "EAP_ROOT is required" && exit 2)
	@test -n "$(GARLTTC_ROOT)" || (echo "GARLTTC_ROOT is required" && exit 2)
	uv run --no-sync python scripts/run_e_jepa_garl_final.py --profile full --stages train freeze --eap-root "$(EAP_ROOT)" --garlttc-root "$(GARLTTC_ROOT)" --output-root "$(FINETUNE_OUTPUT)" --device auto --dry-run

garl-full:
	@test -n "$(EAP_ROOT)" || (echo "EAP_ROOT is required" && exit 2)
	@test -n "$(GARLTTC_ROOT)" || (echo "GARLTTC_ROOT is required" && exit 2)
	uv run --no-sync python scripts/run_e_jepa_garl_final.py --profile full --stages train freeze --eap-root "$(EAP_ROOT)" --garlttc-root "$(GARLTTC_ROOT)" --output-root "$(FINETUNE_OUTPUT)" --device auto --resume

evaluate:
	uv run --no-sync python scripts/evaluate.py --split $(SPLIT) $(METRICS_JSON)

demo:
	uv run --no-sync python scripts/run_demo.py

architecture-validate:
	powershell -ExecutionPolicy Bypass -File scripts/run_evttc_architecture_selection.ps1 -Mode Validate

architecture-smoke:
	powershell -ExecutionPolicy Bypass -File scripts/run_evttc_architecture_selection.ps1 -Mode Smoke -Resume

architecture-screen:
	powershell -ExecutionPolicy Bypass -File scripts/run_evttc_architecture_selection.ps1 -Mode Screen -Resume

eap-analysis:
	uv run --no-sync python scripts/run_eap_evttc_complete.py --profile analysis --objectives both --stages all --resume

eap-full:
	uv run --no-sync python scripts/run_eap_evttc_complete.py --profile full --objectives both --stages all --resume

report:
	uv run --no-sync python scripts/build_report.py --repo-root . --output-dir artifacts/tables/regenerable_report
