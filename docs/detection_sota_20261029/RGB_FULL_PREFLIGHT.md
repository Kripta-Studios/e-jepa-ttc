# Preflight RGB completo de la campaña

Se auditó en CPU y modo de solo lectura toda la población multimedia solicitada:
4.096 tokens D4096 de entrenamiento y los 23.958 frames de las ocho secuencias
de selección completa. No se leyeron labels, anotaciones ni test, y no se usó
GPU. Tampoco se reparó, sobrescribió, excluyó o remuestreó ningún frame.

El runner es `scripts/preflight_detection_rgb_assets.py`. Agrupa por TAR, usa
un worker y un único lector TAR simultáneo, y escribe un recibo JSONL durable
por token y por shard. Cada imagen se extrae por completo, se compara su número
de bytes con el header TAR, se hashea y se carga con Pillow manteniendo
`LOAD_TRUNCATED_IMAGES=False`. La resolución exigida es 1280×720. Cada TAR usado
se hashea secuencialmente y se compara con el inventario público fijado.

## Resultado

| Población | PASS | FAIL |
|---|---:|---:|
| D4096 train | 4.095 | 1 |
| Selección completa | 23.958 | 0 |
| Total | 28.053 | 1 |

Se usaron 107 TARs. Los 107 coinciden en tamaño y SHA-256 con el inventario
público. Un shard se marca FAIL exclusivamente porque contiene el único
miembro que Pillow no puede cargar de forma estricta:

```text
sample_token: OYgB6RGWcq:000000661_000000160
member: rgb/000000661_000000160.png
shard: data/train/OYgB6RGWcq/rgb_shards/rgb-00000.tar
error: OSError: image file is truncated
```

El TAR afectado sí coincide con el inventario fijado: tamaño 1.075.773.440
bytes y SHA-256
`d1f9956ad10e2cbd436625edc734b90de7de0bd7368204bd7e2273208f3d3aa5`.
Por tanto, volver a descargar el mismo artefacto público no cambia este miembro.
El preflight no habilita carga truncada ni decide una exclusión silenciosa.

## Evidencias

Directorio: `output/campaign_data/rgb_full_preflight_v1/`.

| Archivo | SHA-256 |
|---|---|
| `run_manifest.json` | `7c24907f27e069fe5fba906ff309bb6018fbe33d3d3ea90db50aa91c9b25122d` |
| `token_status.jsonl` | `3cbdfa10f77a8ed3701f67584cf0caffed8a8662916b49e422cf147b44cce51d` |
| `shard_status.jsonl` | `e825d256933b63ef34fa6aec163f3a3ffd95eb7acf16c700d7227947c5981724` |
| `summary.json` | `342a00619f12bb57cb97cad6098564541935c153bb375435d9dd05701f721332` |

El ledger por token liga los 28.053 assets legibles a su tamaño y SHA-256
observados. El registro del miembro ilegible conserva token, shard, member,
secuencia, rol y excepción; su tamaño/SHA de bajo nivel se conserva en el recibo
de reproducción específico de la campaña, no se inventa a partir del fallo PIL.

La selección fija no requiere cambios por legibilidad. Para entrenamiento debe
crearse una revisión de manifest explícita y común a todos los modelos antes de
reiniciar desde cero; este preflight no altera el manifest histórico.
