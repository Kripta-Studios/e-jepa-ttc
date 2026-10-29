# Reproducción de la auditoría del plan

Fecha: 2026-10-29. Esta auditoría de solo lectura usa etiquetas y metadatos del split público **train** de eAP. No requiere Torch, GPU, RGB, eventos, checkpoints, acceso a red ni etiquetas test. No ejecuta entrenamiento.

## Entradas fijadas

- Checkout de `Kripta-Studios/phase-ttc` en `24b8f36f6e4aa17c965f9526d8cf486b512a8046`.
- Release HF `NAIL-HNU/eAP-dataset`, revisión `a814a946e60d37f7242358af5f344542177e50c5`.
- Manifest público del repo: `data/manifests/eap_detection_public_files_v1.json`.
- Split del repo: `data/splits/eap_detection_candidate_v1.json`, SHA-256 `91a13c5ae7f6e40ee5fdb4a62da481eba5ffff2b8209ea1110befe3ad7cdc903`.
- Manifest piloto: `docs/detection_local_20261029/evidence/pilot_manifest.json`, SHA-256 `048636cd79e8f5160b97a8056b0e2ed6f78846f4bbb365b77efa5884766be159`.

El directorio que se pasa como `--metadata-root` debe contener `data/train.parquet` y `data/train/<sequence_id>/labels.parquet` para cada una de las cuarenta secuencias del split. Los archivos no se incluyen en este anexo. El helper comprueba tamaños/hashes contra el manifest y rechaza diferencias.

## Entorno y comando

Entorno utilizado: CPython 3.12.14, NumPy 2.5.4, PyArrow 26.0.0. Git disponible para comprobar el checkout. Se utilizó un entorno de auditoría separado del entorno PHASE; no se cambió ningún entorno de entrenamiento.

```bash
python audit_pilot_population_readonly.py \
  --repo /ruta/phase-ttc-sota-review \
  --metadata-root /ruta/eap_metadata \
  --output /ruta/nueva/pilot_population_diagnostic.json
```

El directorio padre de salida debe existir y el archivo de salida **no** debe existir. Sustituir las tres rutas por las locales; no añadir `--test`. El script no tiene ruta de descarga ni abre `data/test.parquet`.

Además de comprobar HEAD, el helper contrasta cinco fuentes locales con los bytes de `git show <BASE>:<path>` antes de leer/importar su geometría. Los cambios ajenos a esas fuentes no se borran ni se rechazan. Desactiva escritura de bytecode al importar el helper geométrico. El SHA-256 del script entregado es:

`4fddd256b7474166ca9586b9fe01f79deb380dacda76b66687a42032ac6db298`.

## Qué se comprobó realmente

1. Hash/tamaño de las cuarenta etiquetas train y del manifest de medios `data/train.parquet`.
2. Conteos de anotaciones y pares `(sequence_id, track_id)` por rol/clase; no se confunden con objetos físicos independientes.
3. Filtros del piloto sobre sus 512 frames: exactamente 829/1.565 objetos train admitidos y 938/1.936 de selección.
4. Soporte admitido de bicycle/other_vehicle en el split completo bajo el perfil legacy documentado.
5. Los nueve bloques numéricos/de hashes de la salida reproducida coinciden con los del diagnóstico inicial. El helper registra también versiones, hash propio y fuentes.
6. Ruff del helper pasó. No se ha ejecutado aquí la suite GPU ni las 232 pruebas publicadas del detector.

La auditoría raíz produjo por separado `class_support_raw.json`, usando los mismos cuarenta archivos fijados; concuerda en los conteos raw. Ese archivo añade soporte por frames y secuencias. El helper entregado reproduce el recuento raw principal y los filtros del diagnóstico, sin exigir reproducir el formato distinto de ese segundo JSON.

En paralelo se verificaron 49 hashes de fuentes del recibo publicado, el XML de 232 tests ligado por hash y la regeneración de `TRAINING_TABLES.md`. Esto verifica evidencia histórica; no representa una segunda ejecución de entrenamiento, inferencia o tests. También se reprodujo el fixture de ambigüedad inversa del tracker descrito en las notas científicas.

## Límites de la reproducción

El filtro admite el tamaño RGB original documentado 1280×720 y el resize pinhole 640×384. Reproduce las fórmulas del perfil histórico: no valida píxeles, distorsión residual, calibración física, disponibilidad de medios ni scorer oficial. La comparación original↔HF y los overlays del piloto se inspeccionaron en sus recibos; no se volvieron a ejecutar sobre imágenes.

No se recalculó el AP desde todas las cajas ni se abrió el checkpoint pesado; no estaban disponibles en este entorno. El recuento de clases no demuestra independencia de adquisiciones ni elimina la exposición previa de TRAIN40. Estos límites permanecen en el plan.

## Integridad del paquete

`SHA256SUMS.txt` incluye el plan, el helper, los JSON, notas y revisiones. El manifest no se incluye a sí mismo. Desde la carpeta extraída, en un sistema que tenga `sha256sum`:

```bash
sha256sum -c SHA256SUMS.txt
```

El plan fue revisado independientemente con SHA-256 `d63065f341594613554aecb48219b4bbc967aed2fd5628d3c6bd599cce7013b7`. Los informes de revisión conservan los hallazgos iniciales y el cierre de las correcciones; revisar el documento no equivale a superar sus gates futuros.
