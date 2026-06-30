.PHONY: setup test lint smoke-data scan-data validate-data index-data split-data train-baseline baseline-trivial baseline-geometric baseline-causal-geometry baseline-event-rate cache-voxel pretrain-jepa train-tiny-cnn train-tiny-cnn-lowlabel train-tiny-cnn-jepa train-tiny-cnn-jepa-lowlabel train-tiny-cnn-jepa-probe clean

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

baseline-causal-geometry:
	uv run --no-sync e-jepa-ttc baseline causal-geometry --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --output artifacts/metrics/causal_geometry_baseline.json --derivative-window 15

baseline-event-rate:
	uv run --no-sync e-jepa-ttc baseline event-rate --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --index data/cache/evttc_index.json --output artifacts/metrics/event_rate_baseline.json

cache-voxel:
	uv run --no-sync e-jepa-ttc cache voxel --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --index data/cache/evttc_index.json --output artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --width 160 --height 90 --bins 5 --no-normalize --metadata-channels

pretrain-jepa:
	uv run --no-sync e-jepa-ttc pretrain jepa --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7 --epochs 160 --batch-size 64 --learning-rate 0.0005 --seed 7 --device auto --pretrain-splits train --validation-splits validation --temporal-horizons-ms 20 60 100 240 500 --max-target-slop-ms 10 --variance-weight 1.0 --min-std 0.05

train-tiny-cnn:
	uv run --no-sync e-jepa-ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_seed7 --epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto

train-tiny-cnn-lowlabel:
	uv run --no-sync e-jepa-ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_seed7_frac10 --epochs 80 --batch-size 32 --learning-rate 0.0003 --seed 7 --device auto --train-fraction 0.1

train-tiny-cnn-jepa:
	uv run --no-sync e-jepa-ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_temporal_jepa_seed7 --epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto --pretrained-encoder artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7/jepa_encoder_best.pt

train-tiny-cnn-jepa-lowlabel:
	uv run --no-sync e-jepa-ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_temporal_jepa_seed7_frac10 --epochs 80 --batch-size 32 --learning-rate 0.0003 --seed 7 --device auto --pretrained-encoder artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7/jepa_encoder_best.pt --train-fraction 0.1

train-tiny-cnn-jepa-probe:
	uv run --no-sync e-jepa-ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_temporal_jepa_probe_seed7 --epochs 80 --batch-size 96 --learning-rate 0.001 --seed 7 --device auto --pretrained-encoder artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7/jepa_encoder_best.pt --freeze-encoder

clean:
	python -m compileall -q src tests

