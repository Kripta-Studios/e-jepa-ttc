# Detection SOTA evidence — current_v1

Bundle público, pequeño y regenerable de evidencia local. `results.json` contiene los escalares extraídos; `manifest.json` enlaza cada archivo empaquetado con el path, tamaño y SHA-256 de su fuente. Los JSON grandes usan gzip determinista (`mtime=0`).

No contiene datasets, RGB/eventos crudos, checkpoints ni estados de optimizer. No acredita una puntuación eAP oficial, SOTA ni generalización TTC.

`source_sha256` acredita los bytes locales originales. Cuando `encoding` declara redacción de rutas, esa transformación no es reversible; la verificación pública recalcula `packaged_sha256` y deriva resultados del payload redactado.

Regenerar y verificar:

```powershell
python scripts/build_detection_sota_evidence.py
python scripts/build_detection_sota_evidence.py --check
python scripts/build_detection_sota_evidence.py --from-bundle --check
```
