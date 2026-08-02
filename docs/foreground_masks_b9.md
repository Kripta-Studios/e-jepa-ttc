# B9: máscaras foreground (soporte acotado)

`src/e_jepa_ttc/data/foreground_masks.py` implementa soporte independiente para
targets foreground. No está conectado a una caché ni a un runner. Una máscara es
exclusivamente un **target de supervisión durante entrenamiento**: nunca es una
entrada del modelo ni forma parte de inferencia.

## Estado material auditado

En la auditoría local del 2026-08-02, el release oficial estaba en el commit
`256661242b8a7f5e56aa3c1c02348b30f6e89de6`. El parquet público de train contenía
88.744 filas, 177.488 referencias `mask_paths` y 64.629 rutas únicas. Ninguna de
esas rutas existía bajo los roots locales auditados (`E:\GarlTTC_dataset`,
`E:\eAP_dataset`, `E:\Garl-TTC`). El release tampoco contiene el checkpoint SAM
`.pth` que espera su utilidad `segment_anything`.

Sí existe un snapshot SAM ViT-L descargado desde Hugging Face en la caché local.
Se validó con un forward promptable de `transformers`, pero ese backend no es un
generador automático de propuestas y no se conecta todavía a este módulo. No se
debe presentar ese smoke como evidencia de foreground ni como soporte bbox-free.

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
ablación documentada. No se usará SAM para convertir cajas GT en una falsa evidencia
bbox-free, ni se usará DINOv3/SAM para seleccionar ventanas SSL-Pure con TTC, cajas,
categorías, depth, 3D o labels futuros. Hasta ejecutar P2 con esas auditorías, no se
afirmará que la integración sea SOTA.

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
