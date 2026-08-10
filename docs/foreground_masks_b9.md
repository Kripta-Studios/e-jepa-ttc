# B9: máscaras foreground (soporte acotado)

`src/e_jepa_ttc/data/foreground_masks.py` implementa soporte independiente para
targets foreground. No está conectado a una caché ni a un runner. Una máscara es
exclusivamente un **target de supervisión durante entrenamiento**: nunca es una
entrada del modelo ni forma parte de inferencia.

## Estado material auditado

La auditoría v2 del 2026-08-10, ejecutada desde el commit publicado `4f5cc46`,
verificó el parquet público de train completo: 88.744 filas, 177.488 referencias
`mask_paths` y 64.629 pares secuencia/ruta únicos. Resolvió cero ficheros bajo seis
roots explícitos (`GarlTTC_dataset`, sus datos, `eAP_dataset`, `data`, `data/train`
y el release). No hay máscaras oficiales materiales disponibles.

La misma auditoría comprobó sin extraer imágenes los 64.629 miembros RGB únicos en
135 TAR: cobertura exacta, cero shards o members ausentes. SAM ViT-L revisión
`6851e0441005b0fb96f2cc4dfac472f3d1b14af1` tiene config, SamImageProcessor,
licencia Apache-2.0 y pesos SHA-256
`a57e1b13cd1545938dfcbc9fb26df7f60de6650237a9383382a874a623564b81`.
DINOv3 ConvNeXt-Tiny también está autocontenido y licenciado localmente. El JSON
firmado es `artifacts/metrics/garl_foreground_resource_audit_v2.json`, identidad
`6e910ec2f389ea8b50c7f0230214217ce7bdcc5bef696712d766637b27f1e246`.

La auditoría no importó Transformers ni cargó pesos. Posteriormente se materializó
el extra `multimodal` desde `uv.lock` y se ejecutó un smoke bbox-prompt real en GPU,
sin descargar pesos. Pasó con `0.4207 s` de inferencia, `1691.39 MiB` peak VRAM,
máscara finita de fracción `.06630` e IoU interno `1.0`. El resultado firmado
`be097e6c…2af5e9` demuestra factibilidad, no calidad de máscara ni mejora TTC.

Un segundo audit preregistrado usó cuatro posiciones deterministas por cada una de
las nueve secuencias train: 36 pares/72 endpoints, sin leer TTC. Pasó todos los
gates: bbox–mask IoU mediana `.5761`, cobertura bbox `.5960`, score interno p10
`.9297`, una degenerada, correlación del cambio de área `.6471` y signo `.8286`.
Esto justifica materialización train-only, pero sigue sin ser una comparación contra
GT segmentation. Identidad `e413337b…b58138`; CSV `226532b5…3ccd65`.

Por tanto, esto **no constituye una integración foreground ni evidencia de
entrenamiento**. `OfficialMaskPathResolver.require()` falla explícitamente y exige
mantener la supervisión desactivada cuando el fichero no es material o la resolución
es ambigua.

## SAM/DINOv3 y comparación con Garl-TTC

SAM y DINOv3 no son un requisito para la comparación multimodal P0. El release de
Garl-TTC recibe un ROI construido con la caja del protocolo; sustituir ese ROI por
una propuesta de SAM/DINOv3 cambiaría la tarea y no permitiría atribuir una mejora al
modelo TTC. La comparación primaria debe conservar P0 y declarar explícitamente el
uso de `P0_oracle_bbox_roi`.

DINOv3 es un extractor visual y no un detector de objetivos; SAM en
`transformers` es promptable y tampoco resuelve por sí solo la detección, la
selección del objetivo ni su asociación temporal. Un experimento bbox-free tendría
que ser un protocolo P2 separado con propuestas automáticas, selector sin cajas,
tracking/asociación, política de objetos múltiples y métricas de recall de objetivo
antes de medir TTC. Sus resultados no se mezclarán con P0.

Los snapshots HF locales pueden usarse como teachers visuales congelados en una
ablación documentada. Para A3 se permite bbox prompt **solo durante train**, porque
las bbox ya son supervisión oracle declarada; esto no es bbox-free. El brazo se
etiquetará `event-only inference with RGB distillation` y se comparará por separado
con event-only puro. No se generarán pseudo-máscaras en validation/test ni se usará
SAM/DINO para seleccionar ventanas, reparar outliers o ajustar TTC post-hoc.

## Contrato implementado

- resolución trazable de rutas absolutas, `root/mask_path` y
  `root/sequence_id/mask_path`, con lista completa de candidatos;
- compresión binaria RLE determinista en orden de filas;
- square crop y resize 256 por defecto, con la misma rejilla bilinear, padding y
  `align_corners` del release; el modo `official_truncate` reproduce su cast entero;
- adaptador de teacher local image-only con checkpoint y SHA-256;
- factoría SAM automática lazy para el backend `segment_anything`: no importa la
  dependencia ni carga el modelo grande hasta una llamada explícita de generación;
- selección de propuestas suministrada y nombrada por el usuario, sin API de cajas
  GT. No se debe presentar una unión arbitraria de propuestas como foreground GT.

No se descarga ningún checkpoint. El adaptador `segment_anything` no acepta
directamente un snapshot Transformers `model.safetensors`; para usarlo habría que
implementar y auditar una política de propuestas automática sin cajas. Si
`segment_anything` o el checkpoint compatible no están disponibles, la generación
falla de forma explícita.
