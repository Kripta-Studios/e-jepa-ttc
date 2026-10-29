# Evidence bundle current_v2

This public, self-contained bundle verifies final data admission v5 and the RAM/serialization admission. It uses an explicit source allow-list. Local paths are redacted recursively; large ledgers use deterministic gzip (`mtime=0`).

Excluded: raw RGB/events, TAR archives, datasets, checkpoints, optimizer state and large synthetic evaluation payloads. Original-byte SHA-256 values remain attestations in `manifest.json`; each redacted packaged payload has its own verified hash.

Verify without the private `output/` tree:

```powershell
python scripts/build_detection_admission_evidence.py --from-bundle --check
```
